# .cmake-format.py - QuOp_QUISA cmake-format configuration
# See standards/CMAKE_STYLE.md for rationale.

with section("format"):
  tab_size = 2
  use_tabchars = False
  line_width = 120
  max_subgroups_hwrap = 2
  max_pargs_hwrap = 6
  dangle_parens = True
  dangle_align = "prefix"
  min_prefix_chars = 4
  max_prefix_chars = 10
  command_case = "lower"
  keyword_case = "upper"

with section("markup"):
  enable_markup = True
  first_comment_is_literal = False

with section("lint"):
  disabled_codes = []
