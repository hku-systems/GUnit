target triple = "nvptx64-nvidia-cuda"

define void @__rapid_entry_test(ptr %data, i32 %index, ptr %__rapid_context) {
entry:
  %positive = icmp sge i32 %index, 0
  br i1 %positive, label %access, label %local

access:
  %element = getelementptr i32, ptr %data, i32 %index
  %value = load i32, ptr %element, align 4
  %next = add i32 %value, 1
  store i32 %next, ptr %element, align 4
  br label %exit

local:
  %scratch = alloca i32, align 4
  store i32 0, ptr %scratch, align 4
  br label %exit

exit:
  ret void
}
