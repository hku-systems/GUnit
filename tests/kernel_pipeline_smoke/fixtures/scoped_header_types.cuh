#ifndef SCOPED_HEADER_TYPES_CUH
#define SCOPED_HEADER_TYPES_CUH

namespace header_scope {

enum class mode {
  plain = 0,
  scaled = 1,
};

struct outer_box {
  struct params {
    int count;
    float *data;
    mode op;
  };
};

}  // namespace header_scope

#endif  // SCOPED_HEADER_TYPES_CUH
