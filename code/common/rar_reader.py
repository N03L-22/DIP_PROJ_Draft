"""Read the files inside a password-protected RAR archive, one at a time, without extracting to disk.

The UIEB authors publish the archive password on their dataset page. The unrar-cffi package does not
expose a password option, so this uses the same low-level calls it uses itself and answers the library's
"password needed" request. Nothing is run or written outside memory.
"""
from unrar.cffi.unrarlib import (FLAGS_RHDF_DIRECTORY, PyUNRARCALLBACKStub, RARProcessFileW, RARSetCallbackPtr,
                                 RarArchive, BadRarFile, UCM_PROCESSDATA, C_RAR_TEST, C_ERAR_SUCCESS, ffi)

UCM_NEEDPASSWORDW = 4  # message number the library sends when it needs the (wide-character) password


def iter_rar_files(path, password, wanted=None):
    """Yield (name, bytes) for each file in the archive. If `wanted` is a set of names, other files are skipped unread."""
    with RarArchive.open_for_processing(str(path)) as archive:
        for header in archive.iterate_headers():
            name = header.FileNameW
            if header.Flags & FLAGS_RHDF_DIRECTORY or (wanted is not None and name not in wanted):
                header.skip()
                continue
            chunks = []

            def answer(message, p1, p2):
                if message == UCM_PROCESSDATA:        # a block of the decoded file
                    chunks.append(bytes(ffi.buffer(ffi.cast("char *", p1), p2)))
                elif message == UCM_NEEDPASSWORDW:    # copy the password into the buffer the library gives us
                    buffer = ffi.cast("wchar_t *", p1)
                    for position, character in enumerate(password[: p2 - 1]):
                        buffer[position] = character
                    buffer[min(len(password), p2 - 1)] = "\0"
                return 1

            handle = ffi.new_handle(answer)
            RARSetCallbackPtr(header.handle, PyUNRARCALLBACKStub, handle)
            result = RARProcessFileW(header.handle, C_RAR_TEST, ffi.NULL, ffi.NULL)  # "test" decodes and reports the data blocks
            RARSetCallbackPtr(header.handle, ffi.NULL, ffi.NULL)
            if result != C_ERAR_SUCCESS:
                raise BadRarFile(f"{name}: error code {result} (wrong password, or a damaged archive)")
            yield name, b"".join(chunks)
