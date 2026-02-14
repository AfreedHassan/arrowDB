#ifndef ARROW_FILE_LOCK_H
#define ARROW_FILE_LOCK_H

#include <filesystem>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include "arrow/utils/result.h"

namespace arrow {

/// RAII file lock using flock() for single-writer enforcement.
class FileLock {
 public:
  static utils::Result<FileLock> acquire(const std::filesystem::path& dir) {
    std::filesystem::create_directories(dir);
    auto lockPath = dir / ".lock";
    int fd = ::open(lockPath.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd < 0)
      return utils::Status(utils::StatusCode::kIoError, "Cannot open lock file");

    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
      ::close(fd);
      return utils::Status(utils::StatusCode::kIoError,
        "Collection is locked by another process");
    }
    return FileLock(fd, lockPath);
  }

  ~FileLock() { release(); }

  FileLock(FileLock&& o) noexcept : fd_(o.fd_), path_(std::move(o.path_)) { o.fd_ = -1; }
  FileLock& operator=(FileLock&& o) noexcept {
    release();
    fd_ = o.fd_; path_ = std::move(o.path_); o.fd_ = -1;
    return *this;
  }

  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;

 private:
  explicit FileLock(int fd, std::filesystem::path path) : fd_(fd), path_(std::move(path)) {}
  void release() {
    if (fd_ >= 0) { ::flock(fd_, LOCK_UN); ::close(fd_); fd_ = -1; }
  }
  int fd_ = -1;
  std::filesystem::path path_;
};

}  // namespace arrow

#endif  // ARROW_FILE_LOCK_H
