target triple = "nvptx64-nvidia-cuda"

define internal i32 @non_entry_select(i1 %condition, i32 %true_value,
                                      i32 %false_value) {
entry:
  %selected = select i1 %condition, i32 %true_value, i32 %false_value
  ret i32 %selected
}

define void @__rapid_entry_select(i1 %condition, i32 %true_value,
                                  i32 %false_value, ptr %output,
                                  ptr %__rapid_context) {
entry:
  %selected = select i1 %condition, i32 %true_value, i32 %false_value
  store i32 %selected, ptr %output, align 4
  ret void
}
