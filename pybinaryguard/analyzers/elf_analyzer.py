"""ELF binary analyzer with a minimal pure-Python ELF parser.

This module provides ``MinimalELFParser`` -- a zero-dependency ELF reader
that extracts only the metadata PyBinaryGuard needs -- and ``ELFAnalyzer``,
which walks a package's installed files and populates
``PackageBinaryInfo.shared_objects``.
"""

from __future__ import annotations

import logging
import os
import struct
from typing import BinaryIO, Dict, List, Optional, Tuple

from pybinaryguard.analyzers.base import AnalyzerBase
from pybinaryguard.models.enums import Architecture
from pybinaryguard.models.package import PackageBinaryInfo, SharedObjectInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ELF constants
# ---------------------------------------------------------------------------

_ELF_MAGIC = b"\x7fELF"

# EI_CLASS
_ELFCLASS32 = 1
_ELFCLASS64 = 2

# EI_DATA
_ELFDATA2LSB = 1  # little-endian
_ELFDATA2MSB = 2  # big-endian

# Section header types
_SHT_NOTE = 7
_SHT_DYNAMIC = 6
_SHT_DYNSYM = 11
_SHT_STRTAB = 3
_SHT_GNU_VERNEED = 0x6FFFFFFE

# Dynamic tag types
_DT_NULL = 0
_DT_NEEDED = 1
_DT_STRTAB = 5
_DT_STRSZ = 10
_DT_SONAME = 14
_DT_RPATH = 15
_DT_RUNPATH = 29

# Note types
_NT_GNU_BUILD_ID = 3

# e_machine -> Architecture mapping
_EM_TO_ARCH: Dict[int, Architecture] = {
    3: Architecture.I686,       # EM_386
    21: Architecture.PPC64LE,   # EM_PPC64
    22: Architecture.S390X,     # EM_S390
    40: Architecture.ARMV7L,    # EM_ARM
    62: Architecture.X86_64,    # EM_X86_64
    183: Architecture.AARCH64,  # EM_AARCH64
}

# Program header types
_PT_DYNAMIC = 2


class ELFParseError(Exception):
    """Raised when a file is not a valid ELF binary or is truncated."""


class MinimalELFParser:
    """Minimal pure-Python ELF parser.

    Extracts only the subset of ELF metadata needed by PyBinaryGuard:

    * ELF class (32/64), endianness, ``e_machine``
    * ``DT_NEEDED``, ``DT_SONAME``, ``DT_RPATH``, ``DT_RUNPATH``
    * GNU version requirements (``SHT_GNU_VERNEED``)
    * GNU build-id (``NT_GNU_BUILD_ID``)

    The parser reads *only* the sections it needs -- it never loads the
    entire file into memory.

    Parameters
    ----------
    path:
        Absolute path to the ELF file.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._ei_class: int = 0  # 1=32, 2=64
        self._ei_data: int = 0   # 1=LE, 2=BE
        self._e_machine: int = 0
        self._e_type: int = 0

        # Section headers (offset, size, type, name_index, sh_link, sh_info, sh_entsize)
        self._sections: List[Dict[str, int]] = []
        self._shstrtab_offset: int = 0
        self._shstrtab_size: int = 0

        # Program headers
        self._phdrs: List[Dict[str, int]] = []

        # Extracted dynamic entries
        self._dt_needed_offsets: List[int] = []
        self._dt_soname_offset: Optional[int] = None
        self._dt_rpath_offset: Optional[int] = None
        self._dt_runpath_offset: Optional[int] = None
        self._dynstr_offset: int = 0
        self._dynstr_size: int = 0

        # Resolved strings
        self._needed: List[str] = []
        self._soname: Optional[str] = None
        self._rpath: Optional[str] = None
        self._runpath: Optional[str] = None

        # Version requirements: list of (library, version_string)
        self._version_requirements: List[Tuple[str, str]] = []

        # Build-id hex string
        self._build_id: Optional[str] = None

        self._parsed = False

    # ------------------------------------------------------------------
    # Struct helpers
    # ------------------------------------------------------------------

    def _endian_prefix(self) -> str:
        """Return struct byte-order prefix for the ELF endianness."""
        return "<" if self._ei_data == _ELFDATA2LSB else ">"

    def _ptr_fmt(self) -> str:
        """Return the struct format character for an address/offset."""
        return "Q" if self._ei_class == _ELFCLASS64 else "I"

    def _ptr_size(self) -> int:
        return 8 if self._ei_class == _ELFCLASS64 else 4

    def _read_at(self, f: BinaryIO, offset: int, size: int) -> bytes:
        """Seek to *offset* and read exactly *size* bytes."""
        f.seek(offset)
        data = f.read(size)
        if len(data) < size:
            raise ELFParseError(
                f"Truncated read at offset {offset}: expected {size} bytes, got {len(data)}"
            )
        return data

    def _read_string(self, f: BinaryIO, strtab_offset: int, str_offset: int) -> str:
        """Read a null-terminated string from a string table."""
        f.seek(strtab_offset + str_offset)
        chunks: List[bytes] = []
        while True:
            byte = f.read(1)
            if not byte or byte == b"\x00":
                break
            chunks.append(byte)
        return b"".join(chunks).decode("ascii", errors="replace")

    # ------------------------------------------------------------------
    # Header parsing
    # ------------------------------------------------------------------

    def _parse_ident(self, f: BinaryIO) -> None:
        """Parse the 16-byte ELF identification."""
        ident = self._read_at(f, 0, 16)
        if ident[:4] != _ELF_MAGIC:
            raise ELFParseError(f"Not an ELF file: {self._path}")

        self._ei_class = ident[4]
        if self._ei_class not in (_ELFCLASS32, _ELFCLASS64):
            raise ELFParseError(f"Unknown ELF class: {self._ei_class}")

        self._ei_data = ident[5]
        if self._ei_data not in (_ELFDATA2LSB, _ELFDATA2MSB):
            raise ELFParseError(f"Unknown ELF data encoding: {self._ei_data}")

    def _parse_header(self, f: BinaryIO) -> None:
        """Parse the ELF header (after ident)."""
        ep = self._endian_prefix()

        # e_type (2 bytes at offset 16), e_machine (2 bytes at offset 18)
        hdr_common = self._read_at(f, 16, 4)
        self._e_type, self._e_machine = struct.unpack(ep + "HH", hdr_common)

        if self._ei_class == _ELFCLASS64:
            # 64-bit header layout (after e_ident[16]):
            # e_type(2) e_machine(2) e_version(4) e_entry(8) e_phoff(8)
            # e_shoff(8) e_flags(4) e_ehsize(2) e_phentsize(2) e_phnum(2)
            # e_shentsize(2) e_shnum(2) e_shstrndx(2)
            data = self._read_at(f, 16, 48)  # offsets 16..63
            (
                _e_type, _e_machine, _e_version,
                _e_entry, e_phoff, e_shoff,
                _e_flags, _e_ehsize,
                e_phentsize, e_phnum,
                e_shentsize, e_shnum, e_shstrndx,
            ) = struct.unpack(ep + "HHIQQQIHHHHHH", data)
        else:
            # 32-bit header layout (after e_ident[16]):
            # e_type(2) e_machine(2) e_version(4) e_entry(4) e_phoff(4)
            # e_shoff(4) e_flags(4) e_ehsize(2) e_phentsize(2) e_phnum(2)
            # e_shentsize(2) e_shnum(2) e_shstrndx(2)
            data = self._read_at(f, 16, 36)  # offsets 16..51
            (
                _e_type, _e_machine, _e_version,
                _e_entry, e_phoff, e_shoff,
                _e_flags, _e_ehsize,
                e_phentsize, e_phnum,
                e_shentsize, e_shnum, e_shstrndx,
            ) = struct.unpack(ep + "HHIIIIIHHHHHH", data)

        self._e_shoff = e_shoff
        self._e_shentsize = e_shentsize
        self._e_shnum = e_shnum
        self._e_shstrndx = e_shstrndx
        self._e_phoff = e_phoff
        self._e_phentsize = e_phentsize
        self._e_phnum = e_phnum

    # ------------------------------------------------------------------
    # Program headers
    # ------------------------------------------------------------------

    def _parse_program_headers(self, f: BinaryIO) -> None:
        """Parse all program headers to locate the PT_DYNAMIC segment."""
        ep = self._endian_prefix()
        for i in range(self._e_phnum):
            offset = self._e_phoff + i * self._e_phentsize
            if self._ei_class == _ELFCLASS64:
                # Elf64_Phdr: p_type(4) p_flags(4) p_offset(8) p_vaddr(8)
                #             p_paddr(8) p_filesz(8) p_memsz(8) p_align(8)
                data = self._read_at(f, offset, 56)
                (
                    p_type, p_flags, p_offset, p_vaddr,
                    p_paddr, p_filesz, p_memsz, p_align,
                ) = struct.unpack(ep + "IIQQQQQQ", data)
            else:
                # Elf32_Phdr: p_type(4) p_offset(4) p_vaddr(4) p_paddr(4)
                #             p_filesz(4) p_memsz(4) p_flags(4) p_align(4)
                data = self._read_at(f, offset, 32)
                (
                    p_type, p_offset, p_vaddr, p_paddr,
                    p_filesz, p_memsz, p_flags, p_align,
                ) = struct.unpack(ep + "IIIIIIII", data)
            self._phdrs.append({
                "p_type": p_type,
                "p_offset": p_offset,
                "p_vaddr": p_vaddr,
                "p_filesz": p_filesz,
                "p_memsz": p_memsz,
            })

    # ------------------------------------------------------------------
    # Section headers
    # ------------------------------------------------------------------

    def _parse_section_headers(self, f: BinaryIO) -> None:
        """Parse all section headers."""
        ep = self._endian_prefix()

        for i in range(self._e_shnum):
            offset = self._e_shoff + i * self._e_shentsize

            if self._ei_class == _ELFCLASS64:
                # Elf64_Shdr: sh_name(4) sh_type(4) sh_flags(8) sh_addr(8)
                #             sh_offset(8) sh_size(8) sh_link(4) sh_info(4)
                #             sh_addralign(8) sh_entsize(8)
                data = self._read_at(f, offset, 64)
                (
                    sh_name, sh_type, sh_flags, sh_addr,
                    sh_offset, sh_size, sh_link, sh_info,
                    sh_addralign, sh_entsize,
                ) = struct.unpack(ep + "IIQQQQIIqq", data)
            else:
                # Elf32_Shdr: sh_name(4) sh_type(4) sh_flags(4) sh_addr(4)
                #             sh_offset(4) sh_size(4) sh_link(4) sh_info(4)
                #             sh_addralign(4) sh_entsize(4)
                data = self._read_at(f, offset, 40)
                (
                    sh_name, sh_type, sh_flags, sh_addr,
                    sh_offset, sh_size, sh_link, sh_info,
                    sh_addralign, sh_entsize,
                ) = struct.unpack(ep + "IIIIIIIIII", data)

            self._sections.append({
                "sh_name": sh_name,
                "sh_type": sh_type,
                "sh_flags": sh_flags,
                "sh_addr": sh_addr,
                "sh_offset": sh_offset,
                "sh_size": sh_size,
                "sh_link": sh_link,
                "sh_info": sh_info,
                "sh_addralign": sh_addralign,
                "sh_entsize": sh_entsize,
            })

        # Load the section header string table so we can resolve names
        if 0 <= self._e_shstrndx < len(self._sections):
            shstrtab = self._sections[self._e_shstrndx]
            self._shstrtab_offset = shstrtab["sh_offset"]
            self._shstrtab_size = shstrtab["sh_size"]

    def _section_name(self, f: BinaryIO, name_offset: int) -> str:
        """Resolve a section name from the section header string table."""
        if self._shstrtab_offset == 0:
            return ""
        return self._read_string(f, self._shstrtab_offset, name_offset)

    # ------------------------------------------------------------------
    # Dynamic section parsing
    # ------------------------------------------------------------------

    def _parse_dynamic_section(self, f: BinaryIO) -> None:
        """Parse the SHT_DYNAMIC section to extract DT_NEEDED, etc.

        We locate the dynamic string table (DT_STRTAB/DT_STRSZ) and then
        find the .dynstr section that covers that address range.
        """
        ep = self._endian_prefix()
        ptr = self._ptr_fmt()
        ps = self._ptr_size()

        dyn_section = None
        for sec in self._sections:
            if sec["sh_type"] == _SHT_DYNAMIC:
                dyn_section = sec
                break

        if dyn_section is None:
            return

        # Each dynamic entry is two pointer-width values: d_tag, d_un
        entry_size = ps * 2
        fmt = ep + ptr + ptr
        count = dyn_section["sh_size"] // entry_size

        dyn_data = self._read_at(f, dyn_section["sh_offset"], dyn_section["sh_size"])

        dt_strtab_vaddr: int = 0
        dt_strsz: int = 0

        for i in range(count):
            d_tag, d_val = struct.unpack_from(fmt, dyn_data, i * entry_size)
            if d_tag == _DT_NULL:
                break
            elif d_tag == _DT_NEEDED:
                self._dt_needed_offsets.append(d_val)
            elif d_tag == _DT_SONAME:
                self._dt_soname_offset = d_val
            elif d_tag == _DT_RPATH:
                self._dt_rpath_offset = d_val
            elif d_tag == _DT_RUNPATH:
                self._dt_runpath_offset = d_val
            elif d_tag == _DT_STRTAB:
                dt_strtab_vaddr = d_val
            elif d_tag == _DT_STRSZ:
                dt_strsz = d_val

        # Resolve DT_STRTAB virtual address to a file offset by finding the
        # matching section.  The dynamic string table section (usually
        # .dynstr) has type SHT_STRTAB and its sh_addr matches DT_STRTAB.
        # If we can't resolve via sections, try program headers.
        self._dynstr_offset = 0
        self._dynstr_size = dt_strsz

        if dt_strtab_vaddr != 0:
            # First try: match a STRTAB section by virtual address
            for sec in self._sections:
                if sec["sh_type"] == _SHT_STRTAB and sec["sh_addr"] == dt_strtab_vaddr:
                    self._dynstr_offset = sec["sh_offset"]
                    if dt_strsz == 0:
                        self._dynstr_size = sec["sh_size"]
                    break

            # Fallback: use the linked section from the dynamic section header
            if self._dynstr_offset == 0 and dyn_section["sh_link"] < len(self._sections):
                linked = self._sections[dyn_section["sh_link"]]
                self._dynstr_offset = linked["sh_offset"]
                if self._dynstr_size == 0:
                    self._dynstr_size = linked["sh_size"]

            # Last resort: compute offset from PT_LOAD segments
            if self._dynstr_offset == 0:
                self._dynstr_offset = self._vaddr_to_offset(dt_strtab_vaddr)

    def _vaddr_to_offset(self, vaddr: int) -> int:
        """Convert a virtual address to a file offset using program headers.

        Iterates over program headers to find a LOAD segment that contains
        *vaddr* and computes the corresponding file offset.  Returns 0 if
        no matching segment is found.
        """
        # PT_LOAD = 1
        for phdr in self._phdrs:
            if phdr["p_type"] != 1:  # PT_LOAD
                continue
            seg_vaddr = phdr["p_vaddr"]
            seg_memsz = phdr["p_memsz"]
            if seg_vaddr <= vaddr < seg_vaddr + seg_memsz:
                return phdr["p_offset"] + (vaddr - seg_vaddr)
        return 0

    def _resolve_dynamic_strings(self, f: BinaryIO) -> None:
        """Read actual string values from the dynstr table."""
        if self._dynstr_offset == 0:
            return

        for off in self._dt_needed_offsets:
            self._needed.append(self._read_string(f, self._dynstr_offset, off))

        if self._dt_soname_offset is not None:
            self._soname = self._read_string(f, self._dynstr_offset, self._dt_soname_offset)

        if self._dt_rpath_offset is not None:
            self._rpath = self._read_string(f, self._dynstr_offset, self._dt_rpath_offset)

        if self._dt_runpath_offset is not None:
            self._runpath = self._read_string(f, self._dynstr_offset, self._dt_runpath_offset)

    # ------------------------------------------------------------------
    # GNU version requirements (SHT_GNU_VERNEED)
    # ------------------------------------------------------------------

    def _parse_verneed(self, f: BinaryIO) -> None:
        """Parse .gnu.version_r to extract GLIBC/GLIBCXX version strings.

        The section contains a linked list of ``Elfxx_Verneed`` entries.
        Each entry has a linked list of ``Elfxx_Vernaux`` sub-entries that
        carry the actual version string indices.

        Verneed layout (both 32- and 64-bit have same sizes):
            vn_version (2)  -- always 1
            vn_cnt     (2)  -- number of Vernaux entries
            vn_file    (4)  -- offset into linked string table for library name
            vn_aux     (4)  -- byte offset to first Vernaux
            vn_next    (4)  -- byte offset to next Verneed (0 = last)

        Vernaux layout:
            vna_hash   (4)
            vna_flags  (2)
            vna_other  (2)
            vna_name   (4)  -- offset into linked string table for version string
            vna_next   (4)  -- byte offset to next Vernaux (0 = last)
        """
        ep = self._endian_prefix()

        for sec in self._sections:
            if sec["sh_type"] != _SHT_GNU_VERNEED:
                continue

            # The linked string table is indicated by sh_link
            strtab_sec_idx = sec["sh_link"]
            if strtab_sec_idx >= len(self._sections):
                continue
            strtab_offset = self._sections[strtab_sec_idx]["sh_offset"]

            sec_data = self._read_at(f, sec["sh_offset"], sec["sh_size"])
            pos = 0

            while pos < len(sec_data):
                if pos + 16 > len(sec_data):
                    break

                vn_version, vn_cnt, vn_file, vn_aux, vn_next = struct.unpack_from(
                    ep + "HHIII", sec_data, pos
                )

                lib_name = self._read_string(f, strtab_offset, vn_file)

                # Walk Vernaux entries
                aux_pos = pos + vn_aux
                for _ in range(vn_cnt):
                    if aux_pos + 16 > len(sec_data):
                        break
                    vna_hash, vna_flags, vna_other, vna_name, vna_next = struct.unpack_from(
                        ep + "IHHII", sec_data, aux_pos
                    )
                    ver_str = self._read_string(f, strtab_offset, vna_name)
                    self._version_requirements.append((lib_name, ver_str))

                    if vna_next == 0:
                        break
                    aux_pos += vna_next

                if vn_next == 0:
                    break
                pos += vn_next

    # ------------------------------------------------------------------
    # GNU build-id (SHT_NOTE with name "GNU\0" and type NT_GNU_BUILD_ID)
    # ------------------------------------------------------------------

    def _parse_build_id(self, f: BinaryIO) -> None:
        """Parse .note.gnu.build-id section."""
        ep = self._endian_prefix()

        for sec in self._sections:
            if sec["sh_type"] != _SHT_NOTE:
                continue

            # Check the section name to avoid parsing irrelevant notes
            name = self._section_name(f, sec["sh_name"])
            if name != ".note.gnu.build-id":
                continue

            note_data = self._read_at(f, sec["sh_offset"], sec["sh_size"])
            pos = 0

            while pos + 12 <= len(note_data):
                namesz, descsz, note_type = struct.unpack_from(ep + "III", note_data, pos)
                pos += 12

                # Align name and desc to 4 bytes
                namesz_aligned = (namesz + 3) & ~3
                descsz_aligned = (descsz + 3) & ~3

                if pos + namesz_aligned + descsz_aligned > len(note_data):
                    break

                note_name = note_data[pos:pos + namesz].rstrip(b"\x00")
                pos += namesz_aligned

                if note_name == b"GNU" and note_type == _NT_GNU_BUILD_ID:
                    desc = note_data[pos:pos + descsz]
                    self._build_id = desc.hex()
                    return

                pos += descsz_aligned

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> Dict[str, object]:
        """Parse the ELF file and return extracted metadata.

        Returns
        -------
        dict
            Keys: ``ei_class``, ``endianness``, ``e_machine``,
            ``e_type``, ``needed``, ``soname``, ``rpath``, ``runpath``,
            ``version_requirements``, ``build_id``.

        Raises
        ------
        ELFParseError
            If the file is not a valid ELF binary or is truncated.
        OSError
            If the file cannot be opened.
        """
        with open(self._path, "rb") as f:
            self._parse_ident(f)
            self._parse_header(f)
            self._parse_program_headers(f)
            self._parse_section_headers(f)
            self._parse_dynamic_section(f)
            self._resolve_dynamic_strings(f)
            self._parse_verneed(f)
            self._parse_build_id(f)

        self._parsed = True

        return {
            "ei_class": 64 if self._ei_class == _ELFCLASS64 else 32,
            "endianness": "little" if self._ei_data == _ELFDATA2LSB else "big",
            "e_machine": self._e_machine,
            "e_type": self._e_type,
            "needed": list(self._needed),
            "soname": self._soname,
            "rpath": self._rpath,
            "runpath": self._runpath,
            "version_requirements": list(self._version_requirements),
            "build_id": self._build_id,
        }

    def get_needed(self) -> List[str]:
        """Return the list of DT_NEEDED library names.

        ``parse()`` must be called first.
        """
        return list(self._needed)

    def get_version_requirements(self) -> List[Tuple[str, str]]:
        """Return GNU version requirements as ``(library, version)`` tuples.

        ``parse()`` must be called first.
        """
        return list(self._version_requirements)


# ---------------------------------------------------------------------------
# ELF Analyzer
# ---------------------------------------------------------------------------


class ELFAnalyzer(AnalyzerBase):
    """Analyzer that discovers and parses ELF shared objects in a package.

    For each ``.so`` file found under the package's ``install_path``:

    1. Parse the ELF header and dynamic section with ``MinimalELFParser``.
    2. Populate a ``SharedObjectInfo`` instance.
    3. Set package-level aggregates (``required_glibc``,
       ``target_architecture``, ``required_libraries``).
    """

    name: str = "elf"

    def analyze(self, package_info: PackageBinaryInfo) -> PackageBinaryInfo:
        """Discover and parse all ELF shared objects under *package_info.install_path*.

        Parameters
        ----------
        package_info:
            The package descriptor to populate.

        Returns
        -------
        PackageBinaryInfo
            The same (mutated) instance.
        """
        install_path = package_info.install_path
        if not install_path or not os.path.isdir(install_path):
            return package_info

        so_paths = self._find_shared_objects(install_path)
        if not so_paths:
            return package_info

        package_info.is_pure_python = False
        all_needed: set = set()  # type: ignore[type-arg]
        max_glibc: Optional[Tuple[int, int]] = None
        arch: Optional[Architecture] = None

        for so_path in so_paths:
            so_info = self._parse_so(so_path)
            if so_info is None:
                continue

            package_info.shared_objects.append(so_info)

            # Aggregate DT_NEEDED
            for lib in so_info.dt_needed:
                all_needed.add(lib)

            # Track maximum GLIBC requirement
            if so_info.required_glibc is not None:
                if max_glibc is None or so_info.required_glibc > max_glibc:
                    max_glibc = so_info.required_glibc

            # Architecture (use the first one found; they should all agree)
            if arch is None and so_info.architecture != Architecture.UNKNOWN:
                arch = so_info.architecture

        package_info.required_libraries = all_needed
        if max_glibc is not None:
            package_info.required_glibc = max_glibc
        if arch is not None:
            package_info.target_architecture = arch

        return package_info

    @staticmethod
    def _find_shared_objects(root: str) -> List[str]:
        """Walk *root* and collect paths to all ``.so`` files."""
        result: List[str] = []
        try:
            for dirpath, _dirnames, filenames in os.walk(root):
                for fname in filenames:
                    if fname.endswith(".so") or ".so." in fname:
                        result.append(os.path.join(dirpath, fname))
        except OSError as exc:
            logger.debug("Error walking %s: %s", root, exc)
        return result

    @staticmethod
    def _parse_so(path: str) -> Optional[SharedObjectInfo]:
        """Parse a single .so file and return its ``SharedObjectInfo``.

        Returns ``None`` if the file cannot be parsed (e.g. not ELF, or I/O
        error).
        """
        try:
            parser = MinimalELFParser(path)
            info = parser.parse()
        except (ELFParseError, OSError) as exc:
            logger.debug("Skipping %s: %s", path, exc)
            return None

        architecture = _EM_TO_ARCH.get(info["e_machine"], Architecture.UNKNOWN)  # type: ignore[arg-type]

        # Compute the maximum GLIBC version from version requirements
        max_glibc: Optional[Tuple[int, int]] = None
        version_req_strs: List[str] = []
        for lib, ver_str in info["version_requirements"]:  # type: ignore[union-attr]
            version_req_strs.append(f"{lib}({ver_str})")
            if ver_str.startswith("GLIBC_"):
                glibc_ver = _parse_glibc_version(ver_str)
                if glibc_ver is not None:
                    if max_glibc is None or glibc_ver > max_glibc:
                        max_glibc = glibc_ver

        # Detect max GLIBCXX
        max_glibcxx: Optional[str] = None
        for lib, ver_str in info["version_requirements"]:  # type: ignore[union-attr]
            if ver_str.startswith("GLIBCXX_"):
                if max_glibcxx is None or ver_str > max_glibcxx:
                    max_glibcxx = ver_str

        try:
            file_size = os.path.getsize(path)
        except OSError:
            file_size = 0

        return SharedObjectInfo(
            path=path,
            filename=os.path.basename(path),
            architecture=architecture,
            elf_class=info["ei_class"],  # type: ignore[arg-type]
            endianness=info["endianness"],  # type: ignore[arg-type]
            dt_needed=info["needed"],  # type: ignore[arg-type]
            dt_soname=info["soname"],  # type: ignore[arg-type]
            dt_rpath=info["rpath"],  # type: ignore[arg-type]
            dt_runpath=info["runpath"],  # type: ignore[arg-type]
            required_glibc=max_glibc,
            required_glibcxx=max_glibcxx,
            gnu_version_requirements=version_req_strs,
            build_id=info["build_id"],  # type: ignore[arg-type]
            file_size=file_size,
        )


def _parse_glibc_version(version_string: str) -> Optional[Tuple[int, int]]:
    """Parse a version string like ``GLIBC_2.17`` into ``(2, 17)``.

    Returns ``None`` if the string cannot be parsed.
    """
    prefix = "GLIBC_"
    if not version_string.startswith(prefix):
        return None
    parts = version_string[len(prefix):].split(".")
    if len(parts) < 2:
        return None
    try:
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return None
