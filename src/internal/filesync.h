#ifndef FILESYNC_H
#define FILESYNC_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>

namespace arrow {
namespace utils {

static inline bool syncFile(const std::string &filePath) {
#ifdef _WIN32
  HANDLE hFile = CreateFileA(filePath.c_str(), GENERIC_WRITE, FILE_SHARE_WRITE,
                             NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hFile == INVALID_HANDLE_VALUE) {
    std::cerr << "Failed to open file for sync: " << filePath << "\n";
    return false;
  }
  bool result = FlushFileBuffers(hFile) != 0;
  CloseHandle(hFile);
  return result;
#else
  int fd = open(filePath.c_str(), O_WRONLY);
  if (fd == -1) {
    std::cerr << "Failed to open file for sync: " << filePath << "\n";
    return false;
  }
#ifdef __APPLE__
  bool result = fcntl(fd, F_FULLFSYNC) == 0;
#else
  bool result = fsync(fd) == 0;
#endif
  close(fd);
  return result;
#endif
}

} // namespace utils
} // namespace arrow

#endif // FILESYNC_H
