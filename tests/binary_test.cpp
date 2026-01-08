#include <gtest/gtest.h>
#include "arrow/utils/binary.h"
#include <filesystem>
#include <sstream>
#include <vector>

using namespace arrow;

class BinaryTest : public ::testing::Test {
protected:
	void SetUp() override {
		testDir = std::filesystem::temp_directory_path() / "arrow_binary_test";
		std::filesystem::create_directories(testDir);
	}

	void TearDown() override {
		if (std::filesystem::exists(testDir)) {
			std::filesystem::remove_all(testDir);
		}
	}

	std::filesystem::path testDir;
	std::string GetTestPath(const std::string& filename) {
		return (testDir / filename).string();
	}
};

TEST_F(BinaryTest, BinaryWriterAndReader_BasicTypes) {
	std::string path = GetTestPath("basic_types.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		int32_t i = -42;
		uint32_t u = 42;
		float f = 3.14159f;
		double d = 2.71828;
		bool b = true;
		char c = 'X';

		writer.write(i);
		writer.write(u);
		writer.write(f);
		writer.write(d);
		writer.write(b);
		writer.write(c);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		int32_t i;
		uint32_t u;
		float f;
		double d;
		bool b;
		char c;

		EXPECT_TRUE(reader.read(i));
		EXPECT_EQ(i, -42);
		EXPECT_TRUE(reader.read(u));
		EXPECT_EQ(u, 42);
		EXPECT_TRUE(reader.read(f));
		EXPECT_NEAR(f, 3.14159f, 1e-5f);
		EXPECT_TRUE(reader.read(d));
		EXPECT_NEAR(d, 2.71828, 1e-10);
		EXPECT_TRUE(reader.read(b));
		EXPECT_TRUE(b);
		EXPECT_TRUE(reader.read(c));
		EXPECT_EQ(c, 'X');
	}
}

TEST_F(BinaryTest, BinaryWriterAndReader_VectorInt) {
	std::string path = GetTestPath("vector_int.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		std::vector<int32_t> v = {1, 2, 3, 4, 5};
		writer.write(v);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::vector<int32_t> v(5);
		EXPECT_TRUE(reader.read(v));
		EXPECT_EQ(v.size(), 5);
		EXPECT_EQ(v[0], 1);
		EXPECT_EQ(v[1], 2);
		EXPECT_EQ(v[2], 3);
		EXPECT_EQ(v[3], 4);
		EXPECT_EQ(v[4], 5);
	}
}

TEST_F(BinaryTest, BinaryWriterAndReader_EmptyVector) {
	std::string path = GetTestPath("empty_vector.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		std::vector<int32_t> v;
		writer.write(v);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::vector<int32_t> v;
		EXPECT_TRUE(reader.read(v));
		EXPECT_EQ(v.size(), 0);
		EXPECT_TRUE(reader.good());
		EXPECT_FALSE(reader.fail());
	}
}

TEST_F(BinaryTest, BinaryWriterAndReader_SingleElementVector) {
	std::string path = GetTestPath("single_element_vector.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		std::vector<float> v = {1.5f};
		writer.write(v);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::vector<float> v(1);
		EXPECT_TRUE(reader.read(v));
		EXPECT_EQ(v.size(), 1);
		EXPECT_NEAR(v[0], 1.5f, 1e-5f);
	}
}

TEST_F(BinaryTest, BinaryWriterAndReader_VectorFloat) {
	std::string path = GetTestPath("vector_float.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		std::vector<float> v = {0.0f, 1.0f, -1.0f, 1e10f, -1e10f};
		writer.write(v);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::vector<float> v(5);
		EXPECT_TRUE(reader.read(v));
		EXPECT_NEAR(v[0], 0.0f, 1e-5f);
		EXPECT_NEAR(v[1], 1.0f, 1e-5f);
		EXPECT_NEAR(v[2], -1.0f, 1e-5f);
		EXPECT_NEAR(v[3], 1e10f, 1e5f);
		EXPECT_NEAR(v[4], -1e10f, 1e5f);
	}
}

TEST_F(BinaryTest, BinaryWriterAndReader_LargeVector) {
	std::string path = GetTestPath("large_vector.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		std::vector<int32_t> v(10000);
		for (size_t i = 0; i < v.size(); ++i) {
			v[i] = static_cast<int32_t>(i);
		}
		writer.write(v);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::vector<int32_t> v(10000);
		EXPECT_TRUE(reader.read(v));
		EXPECT_EQ(v.size(), 10000);
		for (size_t i = 0; i < v.size(); ++i) {
			EXPECT_EQ(v[i], static_cast<int32_t>(i));
		}
	}
}

TEST_F(BinaryTest, BinaryWriterAndReader_String) {
	std::string path = GetTestPath("string.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		writer.writeString("Hello, World!");
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::string s;
		reader.read(s);
		EXPECT_EQ(s, "Hello, World!");
	}
}

TEST_F(BinaryTest, BinaryWriterAndReader_EmptyString) {
	std::string path = GetTestPath("empty_string.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		writer.writeString("");
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::string s = "not empty";
		reader.read(s);
		EXPECT_EQ(s, "");
	}
}

TEST_F(BinaryTest, BinaryWriterAndReader_SingleCharString) {
	std::string path = GetTestPath("single_char_string.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		writer.writeString("A");
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::string s;
		reader.read(s);
		EXPECT_EQ(s, "A");
	}
}

TEST_F(BinaryTest, BinaryWriterAndReader_MultipleStrings) {
	std::string path = GetTestPath("multiple_strings.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		writer.writeString("first");
		writer.writeString("second");
		writer.writeString("third");
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::string s1, s2, s3;
		reader.read(s1);
		reader.read(s2);
		reader.read(s3);
		EXPECT_EQ(s1, "first");
		EXPECT_EQ(s2, "second");
		EXPECT_EQ(s3, "third");
	}
}

TEST_F(BinaryTest, BinaryWriterAndReader_Utf8String) {
	std::string path = GetTestPath("utf8_string.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		writer.writeString("Hello 世界 🌍");
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::string s;
		reader.read(s);
		EXPECT_EQ(s, "Hello 世界 🌍");
	}
}

TEST_F(BinaryTest, BinaryWriterAndReader_LongString) {
	std::string path = GetTestPath("long_string.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		std::string s(10000, 'X');
		writer.writeString(s);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::string s;
		reader.read(s);
		EXPECT_EQ(s.size(), 10000);
		EXPECT_EQ(s, std::string(10000, 'X'));
	}
}

TEST_F(BinaryTest, BinaryReader_ReadPastEnd_ReturnsFalse) {
	std::string path = GetTestPath("read_past_end.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		int32_t v = 42;
		writer.write(v);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		int32_t v;
		EXPECT_TRUE(reader.read(v));
		EXPECT_EQ(v, 42);
		EXPECT_TRUE(reader.good());
		EXPECT_FALSE(reader.eof());

		EXPECT_FALSE(reader.read(v));
		EXPECT_FALSE(reader.good());
		EXPECT_TRUE(reader.eof() || reader.fail());
	}
}

TEST_F(BinaryTest, BinaryReader_VectorReadPastEnd_ReturnsFalse) {
	std::string path = GetTestPath("vector_read_past_end.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		std::vector<int32_t> v = {1, 2};
		writer.write(v);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::vector<int32_t> v(2);
		EXPECT_TRUE(reader.read(v));

		EXPECT_FALSE(reader.read(v));
		EXPECT_FALSE(reader.good());
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::vector<int32_t> v(5);
		EXPECT_FALSE(reader.read(v));
		EXPECT_FALSE(reader.good());
	}
}

TEST_F(BinaryTest, BinaryReader_StringReadPastEnd_ClearsString) {
	std::string path = GetTestPath("string_read_past_end.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		uint64_t size = 100;
		writer.write(size);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::string s = "original";
		reader.read(s);
		EXPECT_EQ(s, "");
	}
}

TEST_F(BinaryTest, BinaryReader_SeekAndTell) {
	std::string path = GetTestPath("seek_tell.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		int32_t v1 = 1, v2 = 2, v3 = 3;
		writer.write(v1);
		writer.write(v2);
		writer.write(v3);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		EXPECT_EQ(reader.tell(), std::streampos(0));

		int32_t v;
		reader.read(v);
		EXPECT_EQ(v, 1);
		EXPECT_EQ(reader.tell(), std::streampos(sizeof(int32_t)));

		reader.seek(0, std::ios::beg);
		EXPECT_EQ(reader.tell(), std::streampos(0));
		reader.read(v);
		EXPECT_EQ(v, 1);

		reader.seek(sizeof(int32_t), std::ios::cur);
		EXPECT_EQ(reader.tell(), std::streampos(2 * sizeof(int32_t)));
		reader.read(v);
		EXPECT_EQ(v, 3);
	}
}

TEST_F(BinaryTest, BinaryReader_SeekEnd) {
	std::string path = GetTestPath("seek_end.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		int32_t v = 42;
		writer.write(v);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		reader.seek(0, std::ios::end);
		EXPECT_EQ(reader.tell(), std::streampos(sizeof(int32_t)));

		int32_t v;
		EXPECT_FALSE(reader.read(v));
	}
}

TEST_F(BinaryTest, BinaryReader_Peek) {
	std::string path = GetTestPath("peek.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		int32_t v = 42;
		writer.write(v);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		EXPECT_NE(reader.peek(), std::istream::traits_type::eof());

		int32_t v;
		reader.read(v);
		EXPECT_EQ(v, 42);
		EXPECT_EQ(reader.peek(), std::istream::traits_type::eof());
	}
}

TEST_F(BinaryTest, BinaryReader_StreamStates) {
	std::string path = GetTestPath("stream_states.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		int32_t v = 42;
		writer.write(v);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		EXPECT_TRUE(reader.good());
		EXPECT_FALSE(reader.fail());
		EXPECT_FALSE(reader.eof());

		int32_t v;
		reader.read(v);
		EXPECT_TRUE(reader.good());
		EXPECT_FALSE(reader.fail());
		EXPECT_FALSE(reader.eof());

		reader.read(v);
		EXPECT_FALSE(reader.good());
		EXPECT_TRUE(reader.fail() || reader.eof());
	}
}

TEST_F(BinaryTest, BinaryWriter_Flush) {
	std::string path = GetTestPath("flush.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		int32_t v = 42;
		writer.write(v);
		writer.flush();
	}

	std::ifstream check(path, std::ios::binary | std::ios::ate);
	EXPECT_GT(check.tellg(), 0);
}

TEST_F(BinaryTest, RoundTrip_ComplexDataStructure) {
	std::string path = GetTestPath("complex_structure.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		writer.writeString("header");
		uint32_t count = 3;
		writer.write(count);

		std::vector<float> v1 = {1.0f, 2.0f, 3.0f};
		std::vector<float> v2 = {4.0f, 5.0f, 6.0f};
		std::vector<float> v3 = {7.0f, 8.0f, 9.0f};

		writer.write(v1);
		writer.write(v2);
		writer.write(v3);

		writer.writeString("footer");
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::string header;
		reader.read(header);
		EXPECT_EQ(header, "header");

		uint32_t count;
		reader.read(count);
		EXPECT_EQ(count, 3);

		std::vector<float> v1(3), v2(3), v3(3);
		reader.read(v1);
		reader.read(v2);
		reader.read(v3);

		EXPECT_NEAR(v1[0], 1.0f, 1e-5f);
		EXPECT_NEAR(v1[1], 2.0f, 1e-5f);
		EXPECT_NEAR(v1[2], 3.0f, 1e-5f);

		EXPECT_NEAR(v2[0], 4.0f, 1e-5f);
		EXPECT_NEAR(v2[1], 5.0f, 1e-5f);
		EXPECT_NEAR(v2[2], 6.0f, 1e-5f);

		EXPECT_NEAR(v3[0], 7.0f, 1e-5f);
		EXPECT_NEAR(v3[1], 8.0f, 1e-5f);
		EXPECT_NEAR(v3[2], 9.0f, 1e-5f);

		std::string footer;
		reader.read(footer);
		EXPECT_EQ(footer, "footer");
	}
}

TEST_F(BinaryTest, RoundTrip_Uint64Values) {
	std::string path = GetTestPath("uint64_values.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		uint64_t v1 = 0;
		uint64_t v2 = UINT64_MAX;
		uint64_t v3 = 1234567890123456789ULL;

		writer.write(v1);
		writer.write(v2);
		writer.write(v3);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		uint64_t v1, v2, v3;
		reader.read(v1);
		reader.read(v2);
		reader.read(v3);

		EXPECT_EQ(v1, 0ULL);
		EXPECT_EQ(v2, UINT64_MAX);
		EXPECT_EQ(v3, 1234567890123456789ULL);
	}
}

TEST_F(BinaryTest, RoundTrip_Int64Values) {
	std::string path = GetTestPath("int64_values.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		int64_t v1 = 0;
		int64_t v2 = INT64_MAX;
		int64_t v3 = INT64_MIN;
		int64_t v4 = -1234567890123456789LL;

		writer.write(v1);
		writer.write(v2);
		writer.write(v3);
		writer.write(v4);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		int64_t v1, v2, v3, v4;
		reader.read(v1);
		reader.read(v2);
		reader.read(v3);
		reader.read(v4);

		EXPECT_EQ(v1, 0LL);
		EXPECT_EQ(v2, INT64_MAX);
		EXPECT_EQ(v3, INT64_MIN);
		EXPECT_EQ(v4, -1234567890123456789LL);
	}
}

TEST_F(BinaryTest, RoundTrip_DoubleValues) {
	std::string path = GetTestPath("double_values.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		double v1 = 0.0;
		double v2 = 1.0;
		double v3 = -1.0;
		double v4 = 1.7976931348623157E+308;
		double v5 = -1.7976931348623157E+308;
		double v6 = 2.2250738585072014E-308;

		writer.write(v1);
		writer.write(v2);
		writer.write(v3);
		writer.write(v4);
		writer.write(v5);
		writer.write(v6);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		double v1, v2, v3, v4, v5, v6;
		reader.read(v1);
		reader.read(v2);
		reader.read(v3);
		reader.read(v4);
		reader.read(v5);
		reader.read(v6);

		EXPECT_EQ(v1, 0.0);
		EXPECT_EQ(v2, 1.0);
		EXPECT_EQ(v3, -1.0);
		EXPECT_DOUBLE_EQ(v4, 1.7976931348623157E+308);
		EXPECT_DOUBLE_EQ(v5, -1.7976931348623157E+308);
		EXPECT_DOUBLE_EQ(v6, 2.2250738585072014E-308);
	}
}

TEST_F(BinaryTest, RoundTrip_BoolValues) {
	std::string path = GetTestPath("bool_values.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		writer.write(true);
		writer.write(false);
		writer.write(true);
		writer.write(false);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		bool v1, v2, v3, v4;
		reader.read(v1);
		reader.read(v2);
		reader.read(v3);
		reader.read(v4);

		EXPECT_TRUE(v1);
		EXPECT_FALSE(v2);
		EXPECT_TRUE(v3);
		EXPECT_FALSE(v4);
	}
}

TEST_F(BinaryTest, RoundTrip_VectorUint8) {
	std::string path = GetTestPath("vector_uint8.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		std::vector<uint8_t> v = {0, 128, 255, 1, 254};
		writer.write(v);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::vector<uint8_t> v(5);
		EXPECT_TRUE(reader.read(v));
		EXPECT_EQ(v[0], 0);
		EXPECT_EQ(v[1], 128);
		EXPECT_EQ(v[2], 255);
		EXPECT_EQ(v[3], 1);
		EXPECT_EQ(v[4], 254);
	}
}

TEST_F(BinaryTest, RoundTrip_VectorInt8) {
	std::string path = GetTestPath("vector_int8.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		std::vector<int8_t> v = {0, -128, 127, 1, -1};
		writer.write(v);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::vector<int8_t> v(5);
		EXPECT_TRUE(reader.read(v));
		EXPECT_EQ(v[0], 0);
		EXPECT_EQ(v[1], -128);
		EXPECT_EQ(v[2], 127);
		EXPECT_EQ(v[3], 1);
		EXPECT_EQ(v[4], -1);
	}
}

TEST_F(BinaryTest, BinaryReader_EmptyStream) {
	std::string path = GetTestPath("empty_stream.bin");
	{
		std::ofstream file(path, std::ios::binary);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		int32_t v;
		EXPECT_FALSE(reader.read(v));
		EXPECT_FALSE(reader.good());

		std::vector<int32_t> vec(5);
		EXPECT_FALSE(reader.read(vec));
		EXPECT_FALSE(reader.good());

		std::string s = "test";
		reader.read(s);
		EXPECT_EQ(s, "");
		EXPECT_FALSE(reader.good());
	}
}

TEST_F(BinaryTest, BinaryWriterAndReader_MixedTypesInSequence) {
	std::string path = GetTestPath("mixed_types.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		writer.writeString("start");
		int32_t i = 42;
		writer.write(i);
		std::vector<float> v = {1.0f, 2.0f, 3.0f};
		writer.write(v);
		double d = 3.14;
		writer.write(d);
		writer.writeString("end");
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::string s1, s2;
		int32_t i;
		std::vector<float> v(3);
		double d;

		reader.read(s1);
		EXPECT_EQ(s1, "start");

		reader.read(i);
		EXPECT_EQ(i, 42);

		reader.read(v);
		EXPECT_NEAR(v[0], 1.0f, 1e-5f);
		EXPECT_NEAR(v[1], 2.0f, 1e-5f);
		EXPECT_NEAR(v[2], 3.0f, 1e-5f);

		reader.read(d);
		EXPECT_NEAR(d, 3.14, 1e-10);

		reader.read(s2);
		EXPECT_EQ(s2, "end");
	}
}

TEST_F(BinaryTest, BinaryWriterAndReader_ComplexMixedTypes) {
	std::string path = GetTestPath("complex_mixed_types.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		writer.writeString("MAGIC_HEADER");
		uint32_t version = 1;
		writer.write(version);

		int64_t timestamp = 1704067200000LL;
		writer.write(timestamp);

		uint64_t numEntries = 3;
		writer.write(numEntries);

		for (uint64_t i = 0; i < numEntries; ++i) {
			writer.writeString("entry_" + std::to_string(i));

			std::vector<float> embedding = {static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2)};
			writer.write(embedding);

			double score = static_cast<double>(i) * 0.1;
			writer.write(score);

			bool active = (i % 2 == 0);
			writer.write(active);
		}

		std::vector<uint8_t> metadata = {0xFF, 0xFE, 0xFD};
		writer.write(metadata);

		writer.writeString("MAGIC_FOOTER");
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		std::string header, footer;
		uint32_t version;
		int64_t timestamp;
		uint64_t numEntries;

		reader.read(header);
		EXPECT_EQ(header, "MAGIC_HEADER");

		reader.read(version);
		EXPECT_EQ(version, 1);

		reader.read(timestamp);
		EXPECT_EQ(timestamp, 1704067200000LL);

		reader.read(numEntries);
		EXPECT_EQ(numEntries, 3);

		for (uint64_t i = 0; i < numEntries; ++i) {
			std::string entryName;
			std::vector<float> embedding(3);
			double score;
			bool active;

			reader.read(entryName);
			EXPECT_EQ(entryName, "entry_" + std::to_string(i));

			reader.read(embedding);
			EXPECT_NEAR(embedding[0], static_cast<float>(i), 1e-5f);
			EXPECT_NEAR(embedding[1], static_cast<float>(i + 1), 1e-5f);
			EXPECT_NEAR(embedding[2], static_cast<float>(i + 2), 1e-5f);

			reader.read(score);
			EXPECT_NEAR(score, static_cast<double>(i) * 0.1, 1e-10);

			reader.read(active);
			EXPECT_EQ(active, (i % 2 == 0));
		}

		std::vector<uint8_t> metadata(3);
		reader.read(metadata);
		EXPECT_EQ(metadata[0], 0xFF);
		EXPECT_EQ(metadata[1], 0xFE);
		EXPECT_EQ(metadata[2], 0xFD);

		reader.read(footer);
		EXPECT_EQ(footer, "MAGIC_FOOTER");
	}
}

TEST_F(BinaryTest, BinaryReader_SeekToBeginningAfterRead) {
	std::string path = GetTestPath("seek_beginning.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		int32_t v = 42;
		writer.write(v);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		int32_t v1, v2;
		reader.read(v1);
		EXPECT_EQ(v1, 42);

		reader.seek(0, std::ios::beg);
		EXPECT_TRUE(reader.read(v2));
		EXPECT_EQ(v2, 42);
	}
}

TEST_F(BinaryTest, BinaryReader_SeekMiddleAndRead) {
	std::string path = GetTestPath("seek_middle.bin");
	{
		std::ofstream file(path, std::ios::binary);
		BinaryWriter writer(std::make_unique<std::ofstream>(std::move(file)));

		int32_t v1 = 1, v2 = 2, v3 = 3;
		writer.write(v1);
		writer.write(v2);
		writer.write(v3);
	}

	{
		std::ifstream file(path, std::ios::binary);
		BinaryReader reader(std::make_unique<std::ifstream>(std::move(file)));

		reader.seek(sizeof(int32_t), std::ios::beg);
		int32_t v;
		reader.read(v);
		EXPECT_EQ(v, 2);
	}
}

TEST_F(BinaryTest, BinaryWriter_StringStream_BasicTypes) {
	std::stringstream ss;
	ss.exceptions(std::ios::binary | std::ios::out | std::ios::in);
	BinaryWriter writer(std::make_unique<std::stringstream>(std::move(ss)));

	writer.write(true);
	writer.write(false);
	writer.writeString("Hello");
	std::vector<int32_t> v = {1, 2, 3};
	writer.write(v);
	double d = 3.14159;
	writer.write(d);
	writer.flush();
}

TEST_F(BinaryTest, BinaryReader_StringStream_BasicTypes) {
	std::string data;

	{
		std::stringstream ss;
		ss.exceptions(std::ios::binary | std::ios::out | std::ios::in);
		ss.write(reinterpret_cast<const char*>("\x01"), 1);
		ss.write(reinterpret_cast<const char*>("\x00"), 1);
		uint64_t len = 5;
		ss.write(reinterpret_cast<const char*>(&len), sizeof(len));
		ss.write("Hello", 5);
		int32_t v1 = 1, v2 = 2, v3 = 3;
		ss.write(reinterpret_cast<const char*>(&v1), sizeof(v1));
		ss.write(reinterpret_cast<const char*>(&v2), sizeof(v2));
		ss.write(reinterpret_cast<const char*>(&v3), sizeof(v3));
		double d = 3.14159;
		ss.write(reinterpret_cast<const char*>(&d), sizeof(d));
		data = ss.str();
	}

	{
		std::stringstream ss(data);
		ss.exceptions(std::ios::binary | std::ios::in);
		BinaryReader reader(std::make_unique<std::stringstream>(std::move(ss)));

		bool b1, b2;
		reader.read(b1);
		reader.read(b2);
		EXPECT_TRUE(b1);
		EXPECT_FALSE(b2);

		std::string s;
		reader.read(s);
		EXPECT_EQ(s, "Hello");

		std::vector<int32_t> v(3);
		reader.read(v);
		EXPECT_EQ(v[0], 1);
		EXPECT_EQ(v[1], 2);
		EXPECT_EQ(v[2], 3);

		double d;
		reader.read(d);
		EXPECT_NEAR(d, 3.14159, 1e-10);
	}
}

TEST_F(BinaryTest, BinaryWriter_StringStream_MultipleStrings) {
	std::stringstream ss;
	ss.exceptions(std::ios::binary | std::ios::out | std::ios::in);
	BinaryWriter writer(std::make_unique<std::stringstream>(std::move(ss)));

	writer.writeString("first");
	writer.writeString("second");
	writer.writeString("third");
	writer.flush();
}

TEST_F(BinaryTest, BinaryReader_StringStream_MultipleStrings) {
	std::string data;

	{
		std::stringstream ss;
		ss.exceptions(std::ios::binary | std::ios::out | std::ios::in);
		uint64_t len1 = 5, len2 = 6, len3 = 5;
		ss.write(reinterpret_cast<const char*>(&len1), sizeof(len1));
		ss.write("first", 5);
		ss.write(reinterpret_cast<const char*>(&len2), sizeof(len2));
		ss.write("second", 6);
		ss.write(reinterpret_cast<const char*>(&len3), sizeof(len3));
		ss.write("third", 5);
		data = ss.str();
	}

	{
		std::stringstream ss(data);
		ss.exceptions(std::ios::binary | std::ios::in);
		BinaryReader reader(std::make_unique<std::stringstream>(std::move(ss)));

		std::string s1, s2, s3;
		reader.read(s1);
		reader.read(s2);
		reader.read(s3);
		EXPECT_EQ(s1, "first");
		EXPECT_EQ(s2, "second");
		EXPECT_EQ(s3, "third");
	}
}

TEST_F(BinaryTest, BinaryWriter_StringStream_LargeVector) {
	std::stringstream ss;
	ss.exceptions(std::ios::binary | std::ios::out | std::ios::in);
	BinaryWriter writer(std::make_unique<std::stringstream>(std::move(ss)));

	std::vector<float> v(1000);
	for (size_t i = 0; i < v.size(); ++i) {
		v[i] = static_cast<float>(i);
	}
	writer.write(v);
	writer.flush();
}

TEST_F(BinaryTest, BinaryReader_StringStream_LargeVector) {
	std::string data;

	{
		std::stringstream ss;
		ss.exceptions(std::ios::binary | std::ios::out | std::ios::in);
		for (size_t i = 0; i < 1000; ++i) {
			float f = static_cast<float>(i);
			ss.write(reinterpret_cast<const char*>(&f), sizeof(f));
		}
		data = ss.str();
	}

	{
		std::stringstream ss(data);
		ss.exceptions(std::ios::binary | std::ios::in);
		BinaryReader reader(std::make_unique<std::stringstream>(std::move(ss)));
		std::vector<float> v(1000);
		EXPECT_TRUE(reader.read(v));
		for (size_t i = 0; i < v.size(); ++i) {
			EXPECT_NEAR(v[i], static_cast<float>(i), 1e-5f);
		}
	}
}

TEST_F(BinaryTest, BinaryWriter_StringStream_EmptyString) {
	std::stringstream ss;
	ss.exceptions(std::ios::binary | std::ios::out | std::ios::in);
	BinaryWriter writer(std::make_unique<std::stringstream>(std::move(ss)));

	writer.writeString("");
	writer.flush();
}

TEST_F(BinaryTest, BinaryReader_StringStream_EmptyString) {
	std::string data;

	{
		std::stringstream ss;
		ss.exceptions(std::ios::binary | std::ios::out | std::ios::in);
		uint64_t len = 0;
		ss.write(reinterpret_cast<const char*>(&len), sizeof(len));
		data = ss.str();
	}

	{
		std::stringstream ss(data);
		ss.exceptions(std::ios::binary | std::ios::in);
		BinaryReader reader(std::make_unique<std::stringstream>(std::move(ss)));
		std::string s = "not empty";
		reader.read(s);
		EXPECT_EQ(s, "");
	}
}

/*
TEST_F(BinaryTest, BinaryWriterAndReader_StringStreamMixedTypes) {
  auto ss = std::make_unique<std::stringstream>();
  std::string data;
	{
		BinaryWriter writer(std::move(ss));
		writer.writeString("header");
		uint32_t count = 2;
		writer.write(count);
		int32_t v1 = 42, v2 = -42;
		writer.write(v1);
		writer.write(v2);
		writer.writeString("footer");
		writer.flush();
    data = writer.str();
	}
  std::cout << "WRITTEN: " << data << std::endl;
  ss->seekg(0, std::ios::beg);
	{
    BinaryReader reader(std::make_unique<std::stringstream>(data));
		std::string header, footer;
		uint32_t count;
		int32_t v1, v2;
		reader.read(header);
		EXPECT_EQ(header, "header");
		reader.read(count);
		EXPECT_EQ(count, 2);
		reader.read(v1);
		reader.read(v2);
		EXPECT_EQ(v1, 42);
		EXPECT_EQ(v2, -42);
		reader.read(footer);
		EXPECT_EQ(footer, "footer");
	}
}
*/
