#ifndef FILESYNC_H
#define FILESYNC_H
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace arrow {
namespace utils {
static inline bool syncFile(std::ofstream &file, const std::string &filePath) {
  file.flush();
#ifdef _WIN32
  HANDLE hFile = CreateFileA(filePath.c_str(), GENERIC_WRITE, FILE_SHARE_WRITE,
                             NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hFile == INVALID_HANDLE_VALUE) {
    std::cerr << "Failed to open file for sync: " << filePath << "\n";
    return false;
  }
  if (!FlushFileBuffers(hFile)) {
    std::cerr << "Failed to sync file: " << filePath << "\n";
    CloseHandle(hFile);
    return false;
  }
  CloseHandle(hFile);
#else
  int fd = -1;
// Try to get fd from stream (non-standard, only works on GCC/Clang)
#ifdef __linux__
  fd = file.rdbuf()->fd();
#endif
  // as fallback open file again for sync
  if (fd == -1) {
    fd = open(filePath.c_str(), O_WRONLY);
  }

  if (fd == -1 || fsync(fd) != 0) {
    std::cerr << "Failed to sync file: " << filePath << "\n";
    if (fd != -1)
      close(fd);
    return 0;
  }
  if (fd != -1)
    close(fd);
#endif
  return 1;
}
} // namespace utils
} // namespace arrow

#endif // FILESYNC_H
