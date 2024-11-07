#include "error.h"
#include <ostream>
#include <sstream>

namespace nyacc {
std::string ErrorInfo::error(std::string_view src) const {
  Location location = this->location;
  location.line++;
  location.col++;
  auto filename = location.file ? *location.file : "<unknown>";
  // Extract the specific line from the source code
  std::ostringstream oss;
  oss << filename << ":" << location.line << ":" << location.col
      << ": error: " << message << "\n";
  std::string error_header = oss.str();
  std::string_view line_str;
  {
    int current_line = 1;
    size_t pos = 0;
    while (pos < src.size()) {
      size_t next_pos = src.find('\n', pos);
      if (next_pos == std::string_view::npos) {
        next_pos = src.size();
      }
      if (current_line == location.line) {
        line_str = src.substr(pos, next_pos - pos);
        break;
      }
      pos = next_pos + 1;
      current_line++;
    }
  }

  std::string error_msg = error_header;

  //   if (!line_str.empty()) {
  // line_str をエラーメッセージに追加
  error_msg += std::string{line_str} + "\n";

  // カラム位置に合わせてインジケータ行を作成
  int num_spaces = location.col - 1;
  std::string indicator(num_spaces, ' ');
  indicator += '^';

  // インジケータ行をエラーメッセージに追加
  error_msg += indicator + "\n";
  //   }

  return error_msg;
};
} // namespace nyacc
