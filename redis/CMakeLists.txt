include_directories(/home/sandeep/utils/DB/redis-stable/src)

set(REDIS_SOURCE_FILES
    redis_api.c)

add_library(fasttext-redis-shared SHARED ${REDIS_SOURCE_FILES})

set_target_properties(fasttext-redis-shared PROPERTIES OUTPUT_NAME fasttext-redis)

target_link_libraries(fasttext-redis-shared fasttext-shared)
