#pragma once
#include "tl/expected.hpp"
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
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

template <typename T> using Result = tl::expected<T, ErrorInfo>;

/// Helper class to build an ErrorInfo with location and message
class ErrorBuilder {
public:
  ErrorBuilder(const Location invoked_loc, const Location error_loc)
      : invoked_loc_(invoked_loc), error_loc_(error_loc), oss_() {
    oss_ << "@" << *invoked_loc_.file << ":" << invoked_loc_.line << ":"
         << invoked_loc_.col << "\n";
  }

  template <typename T> ErrorBuilder &operator<<(const T &value) {
    oss_ << value;
    return *this;
  }

  operator tl::unexpected<ErrorInfo>() const {
    return tl::unexpected<ErrorInfo>(ErrorInfo{oss_.str(), error_loc_});
  }

private:
  Location invoked_loc_;
  Location error_loc_;
  mutable std::ostringstream oss_;
};

template <typename... Args>
tl::unexpected<ErrorInfo> make_error(Location invoke_loc, Location file_loc,
                                     Args &&...args) {
  ErrorBuilder eb{invoke_loc, file_loc};
  if constexpr (sizeof...(args) > 0) {
    (eb << ... << args);
  }
  return tl::unexpected<ErrorInfo>{eb};
}

} // namespace nyacc

#define FATAL(...)                                                             \
  nyacc::make_error(                                                           \
      nyacc::Location{std::make_shared<std::string>(__FILE__), __LINE__, 0},   \
      __VA_ARGS__)

#define EXPECT_EQ(loc, val1, val2)                                             \
  do {                                                                         \
    if ((val1) != (val2)) {                                                    \
      return FATAL(loc, "Expected:\n  ", #val1 " == " #val2, "\nActual:\n  ",  \
                   (val1), " != ", (val2), "\n");                              \
    }                                                                          \
  } while (0)

#define EXPECT_NE(loc, val1, val2)                                             \
  do {                                                                         \
    if ((val1) == (val2)) {                                                    \
      return FATAL(loc, "Expected:\n  ", #val1 " != " #val2, "\nActual:\n  ",  \
                   (val1), " == ", (val2), "\n");                              \
    }                                                                          \
  } while (0)

#define EXPECT_LT(loc, val1, val2)                                             \
  do {                                                                         \
    if (!((val1) < (val2))) {                                                  \
      return FATAL(loc, "Expected:\n  ", #val1 " < " #val2, "\nActual:\n  ",   \
                   (val1), " >= ", (val2), "\n");                              \
    }                                                                          \
  } while (0)

#define EXPECT_LE(loc, val1, val2)                                             \
  do {                                                                         \
    if (!((val1) <= (val2))) {                                                 \
      return FATAL(loc, "Expected:\n  ", #val1 " <= " #val2, "\nActual:\n  ",  \
                   (val1), " > ", (val2), "\n");                               \
    }                                                                          \
  } while (0)

#define EXPECT_GT(loc, val1, val2)                                             \
  do {                                                                         \
    if (!((val1) > (val2))) {                                                  \
      return FATAL(loc, "Expected:\n  ", #val1 " > " #val2, "\nActual:\n  ",   \
                   (val1), " <= ", (val2), "\n");                              \
    }                                                                          \
  } while (0)

#define EXPECT_GE(loc, val1, val2)                                             \
  do {                                                                         \
    if (!((val1) >= (val2))) {                                                 \
      return FATAL(loc, "Expected:\n  ", #val1 " >= " #val2, "\nActual:\n  ",  \
                   (val1), " < ", (val2), "\n");                               \
    }                                                                          \
  } while (0)

#define EXPECT_TRUE(loc, condition)                                            \
  do {                                                                         \
    if (!(condition)) {                                                        \
      return FATAL(loc, "Expected:\n  ", #condition " to be true",             \
                   "\nActual:\n  ", #condition " is false\n");                 \
    }                                                                          \
  } while (0)

#define EXPECT_FALSE(loc, condition)                                           \
  do {                                                                         \
    if (condition) {                                                           \
      return FATAL(loc, "Expected:\n  ", #condition " to be false",            \
                   "\nActual:\n  ", #condition " is true\n");                  \
    }                                                                          \
  } while (0)
