target triple = "nvptx64-nvidia-cuda"

define void @__rapid_entry_spilled(ptr %data, i32 %index, ptr %__rapid_context) {
entry:
  %data.addr = alloca ptr, align 8
  store ptr %data, ptr %data.addr, align 8
  %data.reload = load ptr, ptr %data.addr, align 8
  %element = getelementptr i32, ptr %data.reload, i32 %index
  %value = load i32, ptr %element, align 4
  ret void
}
