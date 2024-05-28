#pragma once

template <typename DataType>
DataType to_type(char* str) {
	if constexpr (std::is_same_v<DataType, char>) {
		return *str;
	} else if constexpr(std::is_floating_point_v<DataType>) {
		return static_cast<DataType>(std::atof(str));
	} else if constexpr(std::is_integral_v<DataType> && std::is_signed_v<DataType>) {
		return static_cast<DataType>(std::atol(str));
	} else if constexpr(std::is_integral_v<DataType> && !std::is_signed_v<DataType>) {
		return static_cast<DataType>(std::stoul(str));
  }
}

struct arg_parser {
	int argc;
	char** argv;
	
	arg_parser(int argc_, char** argv_) : argc(argc_), argv(argv_) {}

	template <typename DataType>
  DataType get(std::string_view arg, DataType def) {
		char* const* begin = argv;
		char* const* end = argv + argc;
		char* const* itr = std::find(begin, end, arg);
		
		if ((itr != end) && (itr++ != end)) {
			return to_type<DataType>(*itr);
		}
		
		return def;
	}

  bool get(std::string_view arg) {
		char* const* begin = argv;
		char* const* end = argv + argc;
		return (std::find(begin, end, arg) != end);
	}
};
