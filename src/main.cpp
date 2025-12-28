#include <iostream>
#include <string>
#include "../includes/arrowdb.h"
#include "../includes/utils.h"

typedef enum {
  META_COMMAND_SUCCESS,
  META_COMMAND_UNRECOGNIZED_COMMAND
} MetaCommandResult;

typedef enum { PREPARE_SUCCESS, PREPARE_UNRECOGNIZED_STATEMENT } PrepareResult;


class InputBuffer {
	public:
		std::string buffer;
		void read() {
			if (!std::getline(std::cin, buffer)) {
				if (buffer[0] == '.') {
					switch (doMetaCommand()) {
						case (META_COMMAND_SUCCESS):
							printf("SUCCESS");
						case (META_COMMAND_UNRECOGNIZED_COMMAND):
							printf("Unrecognized command '%s'\n", buffer.c_str());
					}
				}
			}
		}
		MetaCommandResult doMetaCommand() {
			return MetaCommandResult::META_COMMAND_SUCCESS;
		}
};


void prompt() {
    std::cout << "db > ";
}

int repl() {
    InputBuffer input;
    while (true) {
        prompt();
        input.read();

        if (input.buffer == ".exit") {
            return EXIT_SUCCESS;
        } else {
            std::cout << "Unrecognized command '" << input.buffer << "'.\n";
        }
    }
}


using namespace arrow;

int main() {
	CollectionConfig cfg("my_collection", 768, DistanceMetric::Cosine, DataType::Float16);

	Collection collection(cfg);

	size_t dim = collection.dimension();

	std::cout << "Collection '" << collection.name() << "' created with dimension " << dim << ".\n";
}
