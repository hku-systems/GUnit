target triple = "nvptx64-nvidia-cuda"

%struct.Config = type { i32, ptr }

define void @__rapid_entry_nested(ptr %output, ptr byval(%struct.Config) align 8 %config, ptr %__rapid_context) {
entry:
  %data.field = getelementptr inbounds %struct.Config, ptr %config, i32 0, i32 1
  %data = load ptr, ptr %data.field, align 8
  %value = load i32, ptr %data, align 4
  store i32 %value, ptr %output, align 4
  ret void
}
