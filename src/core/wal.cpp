#include "arrow/wal.h"
#include "arrow/utils/crc32.h"
#include <fstream>
#include <iostream>
#include <variant>
#include <vector>
#include <cstring>

namespace arrow {

// Helper function to compute file-level CRC32 contribution of an entry
// This batches small fields together to minimize function call overhead
// Returns CRC32 computed over entry data starting from initialCrc
static uint32_t computeEntryFileCrc32(const WAL::Entry& entry, uint32_t initialCrc = 0) {
  uint32_t crc = initialCrc;
  
  // Batch small fields together: type, id, dimension (total ~16 bytes on most systems)
  struct {
    WAL::EntryType type;
    VectorID id;
    uint32_t dimension;
  } headerFields = {entry.type, entry.id, entry.dimension};
  
  crc = utils::crc32(&headerFields, sizeof(headerFields), crc);
  
  // Add embedding data if present
  if (!entry.embedding.empty()) {
    crc = utils::crc32(entry.embedding.data(), 
                      entry.embedding.size() * sizeof(float), crc);
  }
  
  // Include entry's own CRC32
  uint32_t entryCrc32 = entry.computeCrc32();
  crc = utils::crc32(&entryCrc32, sizeof(entryCrc32), crc);
  
  return crc;
}
	uint8_t WAL::Header::detectEndianness() {
		// Use a test value to detect endianness
		// 0x01020304 in memory:
		// - Little-endian: 04 03 02 01
		// - Big-endian:    01 02 03 04
		uint32_t test = 0x01020304;
		uint8_t* bytes = reinterpret_cast<uint8_t*>(&test);
		return (bytes[0] == 0x04) ? 0 : 1;  // 0 = little-endian, 1 = big-endian
	}
	
	bool WAL::Header::isEndiannessCompatible() const {
		return endianness == detectEndianness();
	}

	void WAL::Header::write(BinaryWriter& w) const {
		w.write(magic);
		w.write(version);
		w.write(endianness);
		w.write(crc32);
	}
	
	void WAL::Header::read(BinaryReader& r) {
		// Read header fields - if any read fails, header is invalid
		if (!r.read(magic) || !r.read(version) || !r.read(endianness) || !r.read(crc32)) {
			// Set magic to invalid value to signal read failure
			magic = 0;
		}
	}
	
	void WAL::Header::print() const {
		std::cout << "WAL Header:\n";
		std::cout << "  Magic: 0x" << std::hex << magic << std::dec << "\n";
		std::cout << "  Version: " << version << "\n";
		std::cout << "  Endianness: " << (endianness == 0 ? "little-endian" : "big-endian");
		if (!isEndiannessCompatible()) {
			std::cout << " (WARNING: file endianness differs from host)";
		}
		std::cout << "\n";
		std::cout << "  CRC32: 0x" << std::hex << crc32 << std::dec << "\n";
		std::cout << "\n";
	}
	WAL::Entry::Entry(BinaryReader& r) { 
		read(r); 
	}
	WAL::Entry::Entry(std::ifstream &inFile) { 
		BinaryReader r(inFile);
		read(r); 
	}

	uint32_t WAL::Entry::computeCrc32() const {
		// Compute CRC32 over all entry data in a single optimized pass
		// This matches the exact order and layout of data written to file
		uint32_t crc = 0;
		
		// Compute CRC32 incrementally over each field in write order
		// Using direct pointer/size calls avoids template wrapper overhead
		crc = utils::crc32(&type, sizeof(type), crc);
		crc = utils::crc32(&id, sizeof(id), crc);
		crc = utils::crc32(&dimension, sizeof(dimension), crc);
		if (!embedding.empty()) {
			// For the embedding vector, compute CRC32 directly over the data buffer
			crc = utils::crc32(embedding.data(), embedding.size() * sizeof(float), crc);
		}
		
		return crc;
	}

	WAL::Status WAL::Entry::write(BinaryWriter& w) const {
		assert(dimension == embedding.size());
		w.write(type);
		w.write(id);
		w.write(dimension);
		w.write(embedding);
		
		// Compute CRC32 of the entry data in a single optimized pass
		uint32_t computedCrc = computeCrc32();
		
		// Write CRC32 checksum
		w.write(computedCrc);
		
		return Status::SUCCESS;
	}

	WAL::Status WAL::Entry::read(BinaryReader& r) {
		// Read entry fields with error checking
		if (!r.read(type) || !r.read(id) || !r.read(dimension)) {
			std::cerr << "Failed to read entry header fields\n";
			return Status::FAILURE;
		}
		
		embedding.resize(dimension);
		if (!r.read(embedding)) {
			std::cerr << "Failed to read entry embedding data\n";
			return Status::FAILURE;
		}
		
		if (dimension != embedding.size()) {
			std::cerr << "Entry dimension mismatch: expected " << dimension 
			          << ", got " << embedding.size() << "\n";
			return Status::FAILURE;
		}
		
		// Read stored CRC32
		uint32_t storedCrc;
		if (!r.read(storedCrc)) {
			std::cerr << "Failed to read entry CRC32\n";
			return Status::FAILURE;
		}
		
		// Compute CRC32 of the read data in a single optimized pass
		uint32_t computedCrc = computeCrc32();
		
		// Verify CRC32 matches
		if (storedCrc != computedCrc) {
			std::cerr << "CRC32 mismatch: stored=" << storedCrc 
			          << ", computed=" << computedCrc << "\n";
			return Status::FAILURE;
		}
		
		return Status::SUCCESS;
	}

	std::string WAL::Entry::typeToString() const {
		switch (type) {
			case WAL::EntryType::INSERT: return "INSERT";
			case WAL::EntryType::DELETE: return "DELETE";
			case WAL::EntryType::UPDATE: return "UPDATE";
			default: return "UNKNOWN";	
		}
	}

	utils::json WAL::Entry::toJson() const {
		json j = json::object();
		j["type"] = typeToString();
		j["id"] = id;
		j["dimension"] = dimension;
		j["embedding"] = embedding;
		return j;
	}

	void WAL::Entry::print() const {
		std::cout << toJson().dump(2) << "\n";
	}


  WAL::Status WAL::log(const WAL::Entry &entry, std::string pathParam, bool reset) {
    namespace fs = std::filesystem;
    fs::path path = walPath_;

    // Use provided path or default
    if (pathParam != "") {
      path = fs::path(pathParam);
    }

    // Create directory (and parent directories) if it doesn't exist
    if (!fs::exists(path)) {
      try {
        fs::create_directories(path);
      } catch (const std::exception& e) {
        std::cerr << "Failed to create WAL directory: " << path << " - " << e.what() << "\n";
        return Status::FAILURE;
      }
    } else if (!fs::is_directory(path)) {
      std::cerr << "WAL path exists but is not a directory: " << path << "\n";
      return Status::FAILURE;
    }
    const std::string filename = "db.wal";
    fs::path filePath = path / filename;
    
    uint32_t fileCrc32 = 0;
    
    if (reset) {
      // Reset mode: truncate file, write header, write entry, compute and write CRC32
      std::ofstream walFile(filePath, std::ios::out | std::ios::binary | std::ios::trunc);
      
      if (!walFile.is_open()) {
        std::cerr << "Failed to open WAL file: " << filePath << "\n";
        return Status::FAILURE;
      }
      
      BinaryWriter writer(walFile);
      
      // Write header
      Header header;
      header.endianness = Header::detectEndianness();
      header.crc32 = 0;
      header.write(writer);
      
      // Write entry
      Status writeStatus = entry.write(writer);
      if (writeStatus == Status::FAILURE) {
        std::cerr << "Failed to write WAL entry to file: " << filePath << "\n";
        return Status::FAILURE;
      }
      
      // Compute CRC32 of the entry data (for file-level CRC32)
      fileCrc32 = computeEntryFileCrc32(entry);
      
      // Write file CRC32 at the end
      writer.write(fileCrc32);
      walFile.flush();
    } else {
      // Append mode: read existing entries, compute CRC32, append new entry, update CRC32
      std::ifstream readFile(filePath, std::ios::in | std::ios::binary);
      
      if (readFile.is_open()) {
        // Read existing entries to compute cumulative CRC32
        BinaryReader reader(readFile);
        
        // Read header
        Header header;
        header.read(reader);
        if (header.magic == 0x01) {
          // File has valid header, read existing entries
          readFile.seekg(HEADERSIZE, std::ios::beg);
          
          // Read all existing entries and accumulate CRC32
          while (readFile.good() && !readFile.eof()) {
            // Check if we're at the file CRC32 position
            std::streampos currentPos = readFile.tellg();
            readFile.seekg(0, std::ios::end);
            std::streampos fileEnd = readFile.tellg();
            readFile.seekg(currentPos, std::ios::beg);
            
            // If we're at file CRC32 position, skip it (we'll recompute from entries)
            if (fileEnd - currentPos == FILECRC32SIZE) {
              break;
            }
            
            // Try to read an entry
            EntryPtr existingEntry = std::make_unique<Entry>(reader);
            if (existingEntry->read(reader) == Status::SUCCESS) {
              // Accumulate CRC32 from entry data (using efficient helper)
              fileCrc32 = computeEntryFileCrc32(*existingEntry, fileCrc32);
            } else {
              break;
            }
          }
        }
        readFile.close();
      }
      
      // Now append: open in append mode but we need to overwrite the CRC32 at the end
      // So we'll open in read-write mode, seek to before CRC32, write entry, then CRC32
      std::fstream walFile(filePath, std::ios::in | std::ios::out | std::ios::binary);
      
      if (!walFile.is_open()) {
        std::cerr << "Failed to open WAL file: " << filePath << "\n";
        return Status::FAILURE;
      }
      
      // Seek to position before file CRC32 (or to end if file is empty)
      walFile.seekg(0, std::ios::end);
      std::streampos fileSize = walFile.tellg();
      
      // Check if file has at least header + CRC32 (minimum valid WAL file size)
      if (fileSize < HEADERSIZE + FILECRC32SIZE) {
        std::cerr << "File " << filePath << " is too small or doesn't have a CRC32. ";
        std::cerr << "File size: " << fileSize << ", expected at least " 
                  << (HEADERSIZE + FILECRC32SIZE) << " bytes.\n";
        std::cerr << "Please use reset mode to create a new file with CRC32.\n";
        return Status::FAILURE;
      }
      
      // File has CRC32, seek to position before it
      walFile.seekp(static_cast<int>(fileSize) - FILECRC32SIZE, std::ios::beg);
      
      BinaryWriter writer(walFile);
      
      // Write new entry
      Status writeStatus = entry.write(writer);
      if (writeStatus == Status::FAILURE) {
        std::cerr << "Failed to write WAL entry to file: " << filePath << "\n";
        return Status::FAILURE;
      }
      
      // Update file CRC32 with new entry (using efficient helper)
      fileCrc32 = computeEntryFileCrc32(entry, fileCrc32);
      
      // Write updated file CRC32 at the end
      writer.write(fileCrc32);
      walFile.flush();
    }
    
    return Status::SUCCESS;
  }

	std::variant<WAL::Header, WAL::Status> WAL::readHeader(std::string pathParam) const {
    namespace fs = std::filesystem;
    fs::path path = walPath_;

    if (pathParam != "") {
      path = fs::path(pathParam);
    }

    if (!fs::exists(path)) {
      std::cerr << "WAL directory does not exist: " << path << "\n";
      return Status::FAILURE;
    }
    
    if (!fs::is_directory(path)) {
      std::cerr << "WAL path exists but is not a directory: " << path << "\n";
      return Status::FAILURE;
    }

    const std::string filename = "db.wal";
    fs::path filePath = path / filename;
    std::ifstream walFile(filePath, std::ios::in | std::ios::binary);
    
    if (!walFile.is_open()) {
      std::cerr << "Failed to open WAL file: " << filePath << "\n";
      return Status::FAILURE;
    }

    
    // Check if file is empty (no header)
    walFile.seekg(0, std::ios::end);
    if (walFile.tellg() == 0) {
      // Empty file, return default header with current endianness
      Header defaultHeader;
      defaultHeader.endianness = Header::detectEndianness();
      defaultHeader.crc32 = 0;
      return defaultHeader;
    }
    
    // Read header from beginning
    walFile.seekg(0, std::ios::beg);
    BinaryReader r(walFile);
		Header header;
    header.read(r);
    
    // Check if header read succeeded
    if (!r.good() || header.magic == 0) {
      std::cerr << "Failed to read WAL header or header is invalid\n";
      return Status::FAILURE;
    }
    
    // Verify magic number
    if (header.magic != 0x01) {
      std::cerr << "Invalid WAL magic number: " << header.magic << "\n";
      return Status::FAILURE;
    }
    
    // Check endianness compatibility
    if (!header.isEndiannessCompatible()) {
      std::cerr << "WARNING: WAL file endianness (" 
                << (header.endianness == 0 ? "little" : "big")
                << "-endian) differs from host endianness ("
                << (Header::detectEndianness() == 0 ? "little" : "big")
                << "-endian). File may not be readable.\n";
      // Note: We continue reading but data may be corrupted
    }
    
		return header;
  }
std::variant<WAL::Header, WAL::Status> WAL::readHeader(std::ifstream& is) const {
		BinaryReader r(is);
		Header header;
		header.read(r);
		
		// Verify magic number
		if (header.magic != 0x01) {
			std::cerr << "Invalid WAL magic number: " << header.magic << "\n";
			return Status::FAILURE;
		}
		
		// Check endianness compatibility
		if (!header.isEndiannessCompatible()) {
			std::cerr << "WARNING: WAL file endianness (" 
			          << (header.endianness == 0 ? "little" : "big")
			          << "-endian) differs from host endianness ("
			          << (Header::detectEndianness() == 0 ? "little" : "big")
			          << "-endian). File may not be readable.\n";
			// Note: We continue reading but data may be corrupted
		}
		
		return header;
	}

  std::variant<WAL::EntryPtr, WAL::Status> WAL::read(std::string pathParam) {
    namespace fs = std::filesystem;
    fs::path path = walPath_;

    if (pathParam != "") {
      path = fs::path(pathParam);
    }

    if (!fs::exists(path) || !fs::is_directory(path)) {
      std::cerr << "WAL directory does not exist: " << path << "\n";
      return Status::FAILURE;
    }

    const std::string filename = "db.wal";
    fs::path filePath = path / filename;
    std::ifstream walFile(filePath, std::ios::in | std::ios::binary);
    
    if (!walFile.is_open()) {
      std::cerr << "Failed to open WAL file: " << filePath << "\n";
      return Status::FAILURE;
    }
    
    BinaryReader r(walFile);
		Header header;
    
    // Read header if file has content
    walFile.seekg(0, std::ios::end);
    size_t fileSize = walFile.tellg();
    
    // Calculate header size: magic(4) + version(2) + endianness(1) + crc32(4) = 11 bytes
    
    if (fileSize == 0) {
      std::cerr << "WAL file is empty, no entries to read\n";
      return Status::FAILURE;
    }
    
    if (fileSize < HEADERSIZE) {
      std::cerr << "WAL file is too small to contain a valid header\n";
      return Status::FAILURE;
    }
    
    walFile.seekg(0, std::ios::beg);
    header.read(r);
    
    // Verify magic number
    if (header.magic != 0x01) {
      std::cerr << "Invalid WAL magic number: " << header.magic << "\n";
      return Status::FAILURE;
    }
    
    // Check endianness compatibility
    if (!header.isEndiannessCompatible()) {
      std::cerr << "WARNING: WAL file endianness differs from host. Data may be corrupted.\n";
    }
    
    // Check if there's data after the header
    size_t remainingSize = fileSize - HEADERSIZE;
    if (remainingSize == 0) {
      std::cerr << "WAL file contains only header, no entries\n";
      return Status::FAILURE;
    }
    
    // Check if file only has header + CRC32 (no entries)
    if (remainingSize == FILECRC32SIZE) {
      std::cerr << "WAL file contains only header and CRC32, no entries\n";
      return Status::FAILURE;
    }
    
    if (remainingSize < FILECRC32SIZE) {
      std::cerr << "WAL file is too small: remaining size (" << remainingSize 
                << ") is less than file CRC32 size (" << FILECRC32SIZE << ")\n";
      return Status::FAILURE;
    }
    
    // Read first entry
    // Note: Entry constructor calls read(), so we check the result
    EntryPtr pEntry = std::make_unique<Entry>(r);
    
    // Verify the entry was read successfully by checking stream state
    if (!r.good() && !walFile.eof()) {
      std::cerr << "Failed to read entry from WAL file\n";
      return Status::FAILURE;
    }
    
    // Verify entry has valid data (check if dimension matches embedding size)
    if (pEntry->dimension != pEntry->embedding.size()) {
      std::cerr << "Entry dimension mismatch after read\n";
      return Status::FAILURE;
    }
    
    return pEntry;
  }

  std::variant<std::vector<WAL::EntryPtr>, WAL::Status> WAL::readAll(std::string pathParam) const {
    namespace fs = std::filesystem;
    fs::path path = walPath_;

    if (pathParam != "") {
      path = fs::path(pathParam);
    }

    if (!fs::exists(path)) {
      std::cerr << "WAL directory does not exist: " << path << "\n";
      return Status::FAILURE;
    }
    
    if (!fs::is_directory(path)) {
      std::cerr << "WAL path exists but is not a directory: " << path << "\n";
      return Status::FAILURE;
    }

    const std::string& filename = "db.wal";
    fs::path filePath = path / filename;
    std::ifstream walFile(filePath, std::ios::in | std::ios::binary);
    
    if (!walFile.is_open()) {
      std::cerr << "Failed to open WAL file: " << filePath << "\n";
      return Status::FAILURE;
    }
    
    BinaryReader r(walFile);
    
    // Read header if file has content
    walFile.seekg(0, std::ios::end);
    size_t fileSize = walFile.tellg();
    Header header;
    static constexpr size_t headerSize = sizeof(uint32_t) + sizeof(uint16_t) + sizeof(uint8_t) + sizeof(uint32_t);
    
    if (fileSize == 0) {
      // Empty file, return empty entries vector
      return std::vector<EntryPtr>{};
    }
    
    if (fileSize < headerSize) {
      std::cerr << "WAL file is too small to contain a valid header\n";
      return Status::FAILURE;
    }
    
    walFile.seekg(0, std::ios::beg);
    header.read(r);
    
    // Check if header read succeeded
    if (!r.good() || header.magic == 0) {
      std::cerr << "Failed to read WAL header or header is invalid\n";
      return Status::FAILURE;
    }
    
    // Verify magic number
    if (header.magic != 0x01) {
      std::cerr << "Invalid WAL magic number: " << header.magic << "\n";
      return Status::FAILURE;
    }
    
    // Check endianness compatibility
    if (!header.isEndiannessCompatible()) {
      std::cerr << "WARNING: WAL file endianness differs from host. Data may be corrupted.\n";
    }

    std::vector<EntryPtr> entries;
    
    // Position after header
    walFile.seekg(headerSize, std::ios::beg);
    if (!walFile.good()) {
      std::cerr << "Failed to seek past header\n";
      return Status::FAILURE;
    }
    
    // Compute CRC32 of all entries as we read them
    uint32_t computedFileCrc32 = 0;
    
    // Read all entries until we reach file CRC32 position
    // Track file position to detect infinite loops
    std::streampos lastPos = walFile.tellg();
    size_t consecutiveNoProgress = 0;
    
    while (walFile.good() && !walFile.eof()) {
      // Check current position relative to file end
      std::streampos currentPos = walFile.tellg();
      walFile.seekg(0, std::ios::end);
      std::streampos fileEnd = walFile.tellg();
      walFile.seekg(currentPos, std::ios::beg);
      
      // If we're at file CRC32 position, read it and break
      if (fileEnd - currentPos == FILECRC32SIZE) {
        uint32_t storedFileCrc32;
        if (!r.read(storedFileCrc32)) {
          std::cerr << "Failed to read file CRC32\n";
          return Status::FAILURE;
        }
        
        // Verify file CRC32 matches computed CRC32
        if (storedFileCrc32 != computedFileCrc32) {
          std::cerr << "File CRC32 mismatch: stored=" << storedFileCrc32 
                    << ", computed=" << computedFileCrc32 << "\n";
          return Status::FAILURE;
        }
        break;
      }
      
      // Check if we're at EOF before attempting to read entry
      if (walFile.peek() == EOF) {
        std::cerr << "Unexpected EOF before file CRC32\n";
        return Status::FAILURE;
      }
      
      // Try to read an entry (constructor calls read())
      EntryPtr pEntry = std::make_unique<Entry>(r);
      
      // Check if read succeeded by verifying stream state
      if (!r.good() && !walFile.eof()) {
        // Stream error occurred
        std::cerr << "Stream error while reading WAL entry\n";
        return Status::FAILURE;
      }
      
      // Verify entry has valid data
      if (pEntry->dimension != pEntry->embedding.size()) {
        std::cerr << "Entry dimension mismatch after read\n";
        return Status::FAILURE;
      }
      
      // Accumulate CRC32 for file-level checksum (using efficient helper)
      computedFileCrc32 = computeEntryFileCrc32(*pEntry, computedFileCrc32);
      
      // Check if we made progress
      currentPos = walFile.tellg();
      if (currentPos == lastPos) {
        consecutiveNoProgress++;
        if (consecutiveNoProgress > 10) {
          std::cerr << "No progress reading entries, possible infinite loop\n";
          return Status::FAILURE;
        }
      } else {
        consecutiveNoProgress = 0;
        lastPos = currentPos;
      }
      
      entries.push_back(std::move(pEntry));
      
      if (entries.size() > 1000000) {
        std::cerr << "Too many entries read, possible infinite loop\n";
        return Status::FAILURE;
      }
    }

    return entries;
  }

	void WAL::print() const {
		// Load and print header
		auto h = readHeader();
		if (std::holds_alternative<Header>(h)) {
			const auto& header = std::get<Header>(h);
			header.print();
		} else {
			std::cout << "WAL Header: (not available)\n\n";
		}
		
		// Print all entries
		auto entries = readAll();
		if (std::holds_alternative<std::vector<EntryPtr>>(entries)) {
			const auto& entryVec = std::get<std::vector<EntryPtr>>(entries);
			std::cout << "WAL Entries (" << entryVec.size() << "):\n";
			for (const auto& entry : entryVec) {
				entry->print();
			}
		} else {
			std::cout << "No entries found or error reading entries.\n";
		}
	}

	WAL::WAL() {
    namespace fs = std::filesystem;
    fs::path defaultPath = "wal";
    if (!fs::exists(defaultPath)) {
      fs::create_directories(defaultPath);
      std::cout << "Created WAL directory at "
                << fs::absolute(defaultPath) << "\n";
    }
    walPath_ = defaultPath;
  }
} // namespace arrow

