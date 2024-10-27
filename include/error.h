#pragma once
#include <iostream>
#include <memory>
#include <string>

namespace nyacc {

/// Structure definition a location in a file.
struct Location {
  std::shared_ptr<std::string> file; ///< filename.
  int line;                          ///< line number.
  int col;                           ///< column number.
};
struct ErrorInfo {
  std::string message;
  nyacc::Location location;

  std::string error(std::string_view src) const;
};
} // namespace nyacc
// namespace nyacc