/**
 * @file thread_safety_test.cpp
 * @brief Thread safety tests for concurrent embedding calls
 */

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <embedder/embedder.h>
#include <cmath>
#include <iostream>

TEST(ThreadSafety, ConcurrentEmbedCalls) {
    std::cout << "\n=== Testing concurrent embed calls (8 threads, 100 calls each) ===" << std::endl;
    
    Embedder embedder;
    ASSERT_TRUE(embedder.ok()) << "Failed to initialize embedder";
    
    const int num_threads = 8;
    const int calls_per_thread = 100;
    std::vector<std::thread> threads;
    std::vector<std::vector<float>> results(num_threads * calls_per_thread);
    
    auto task = [&](int thread_id) {
        for (int i = 0; i < calls_per_thread; ++i) {
            std::string text = "Test text " + std::to_string(thread_id) + "_" + std::to_string(i);
            results[thread_id * calls_per_thread + i] = embedder.embed(text.c_str());
        }
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(task, t);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Completed " << (num_threads * calls_per_thread) 
              << " embeddings in " << duration << "ms" << std::endl;
    
    int errors = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].size() != 384) {
            std::cerr << "Error: Embedding " << i << " has wrong dimension: " 
                      << results[i].size() << std::endl;
            errors++;
        } else {
            float norm = 0.0f;
            for (float v : results[i]) norm += v * v;
            norm = std::sqrt(norm);
            if (std::abs(norm - 1.0f) > 0.01f) {
                std::cerr << "Error: Embedding " << i << " not normalized: " << norm << std::endl;
                errors++;
            }
        }
    }
    
    EXPECT_EQ(errors, 0) << "Found " << errors << " embedding errors in " 
                        << (num_threads * calls_per_thread) << " concurrent calls";
}

TEST(ThreadSafety, ConcurrentInitialization) {
    std::cout << "\n=== Testing concurrent initialization (4 threads) ===" << std::endl;
    
    std::vector<std::thread> threads;
    std::vector<int> results(4);
    
    auto init_task = [&](int thread_id) {
        Embedder embedder;
        results[thread_id] = embedder.ok() ? 0 : -1;
    };
    
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back(init_task, t);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    int success_count = 0;
    for (int res : results) {
        if (res == 0) success_count++;
    }
    
    std::cout << "Initialization results: " << success_count << "/4 succeeded" << std::endl;
    ASSERT_GT(success_count, 0) << "No initialization succeeded";
}

TEST(ThreadSafety, RapidSequentialCalls) {
    std::cout << "\n=== Testing rapid sequential calls (1000 iterations) ===" << std::endl;
    
    Embedder embedder;
    ASSERT_TRUE(embedder.ok()) << "Failed to initialize embedder";
    
    const int num_calls = 1000;
    std::vector<std::vector<float>> results(num_calls);
    
    for (int i = 0; i < num_calls; ++i) {
        std::string text = "Sequential test " + std::to_string(i);
        results[i] = embedder.embed(text.c_str());
    }
    
    int errors = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].size() != 384) {
            std::cerr << "Error: Embedding " << i << " has wrong dimension" << std::endl;
            errors++;
        }
    }
    
    EXPECT_EQ(errors, 0) << "Found " << errors << " errors in " << num_calls << " sequential calls";
}

TEST(ThreadSafety, StressTest) {
    std::cout << "\n=== Running stress test (4 threads, 500 calls each) ===" << std::endl;
    
    Embedder embedder;
    ASSERT_TRUE(embedder.ok()) << "Failed to initialize embedder";
    
    const int num_threads = 4;
    const int calls_per_thread = 500;
    std::vector<std::thread> threads;
    std::vector<int> errors(num_threads, 0);
    
    auto task = [&](int thread_id) {
        for (int i = 0; i < calls_per_thread; ++i) {
            std::string text = "Stress test " + std::to_string(thread_id) + "_" + std::to_string(i);
            auto emb = embedder.embed(text.c_str());
            if (emb.size() != 384) {
                errors[thread_id]++;
            }
        }
    };
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(task, t);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    int total_errors = 0;
    for (int e : errors) {
        total_errors += e;
    }
    
    std::cout << "Stress test completed with " << total_errors << " errors out of " 
              << (num_threads * calls_per_thread) << " calls" << std::endl;
    
    EXPECT_EQ(total_errors, 0) << "Found errors in stress test";
}
