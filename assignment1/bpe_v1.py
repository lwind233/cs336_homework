import regex as re
from multiprocessing import Pool
import time
from collections import Counter
import os
from typing import BinaryIO
from tqdm import tqdm

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_tokens: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # assert isinstance(split_special_token, bytes), (
    #     "Must represent special token as a bytestring"
    # )

    byte_split_special_tokens = [i.encode('utf-8') for i in split_special_tokens]

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            for split_special_token in byte_split_special_tokens:
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    break
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def remove_special_token(chunk, special_token):
    escaped_token = re.escape(special_token)
    return re.split(f'({special_token})',chunk)

def pre_tokenization(split_chunk):
    ls = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for i in split_chunk:
        if i == '<|endoftext|>':
            ls.append(i)
        else:
            tmp_result = re.findall(PAT,i)
            ls = ls + tmp_result
    return ls

def get_tokens(task,file_path,special_token):
    start, end = task
    with open(file_path ,'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        split_chunk = remove_special_token(chunk,special_token)
        pre_tokens = pre_tokenization(split_chunk)
    return dict(Counter(pre_tokens))

def convert(x):
    if isinstance(x,int):
        return (x,)
    else:
        return x
    
def detect(token,tup):
    for i in range(len(tup)-1):
        e_f = convert(tup[i])
        e_b = convert(tup[i+1])
        if e_f+e_b==token:
            return i
    return False

def tokenizer(input_path,vocab_size,special_tokens):
    start  = time.time()
    num_processes = 100
    # special_token = 
    ## Usage

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, special_tokens)
            
    # The following is a serial implementation, but you can parallelize this 
    # by sending each start/end pair to a set of processes.
    tokens = {}
    tasks = zip(boundaries[:-1], boundaries[1:])

    print('分块区域查找完毕')
    special_token = "|".join([re.escape(special_token) for special_token in special_tokens])
    with Pool(processes=num_processes) as p:
        async_results = [p.apply_async(get_tokens,(i,input_path,special_token)) for i in tasks]
        p.close()
        p.join()

    end = time.time()
    print('预分词时间：{:.4f}s'.format(end-start))

    for async_result in async_results:
        dict_result = async_result.get()
        for key,value in dict_result.items():
            bkey = key.encode('utf-8')
            t_key = tuple(bkey)
            tokens[t_key] = tokens.get(t_key,0) + value

    vocab = {i:tuple(chr(i).encode('utf-8')) for i in range(256)}
    merges = [tuple(chr(i).encode('utf-8')) for i in range(256)]

    merges.append(tuple(special_token.encode('utf-8')))
    vocab[tuple(special_token.encode('utf-8'))] = len(merges)-1
    
    start = time.time()
    progress = tqdm(total=vocab_size, desc="分词器训练进度")
    while(len(vocab)<vocab_size):
        tmp_frequncy = {}
        # tokens是一个词典，键为元组，值为词频
        for t_key,value in tokens.items():
            if t_key in merges:
                # 遇到了该特殊token，直接跳过
                continue

            for i in range(len(t_key)-1):
                # 将元组中的两个相邻元素（元组或整数）合并为一个新的元组
                e_f = convert(t_key[i])
                e_b = convert(t_key[i+1])
                tmp_frequncy[e_f+e_b] = tmp_frequncy.get(e_f+e_b,0) + value

        # 获取词频最高的token
        max_value = max(tmp_frequncy.values())
        max_keys = [k for k, v in tmp_frequncy.items() if v == max_value]
        fin_key = max(max_keys)

        #加入到词汇表里
        merges.append(fin_key)
        vocab[fin_key] = len(merges)-1

        # 对词频表进行更新
        change_dict = {}
        for t_key,value in tokens.items():
            if t_key in merges:
                continue
            else:
                change = False
                tmp = t_key
                while(True):
                    # 判断当前键里有没有最新的高频token
                    result = detect(fin_key,tmp)
                    if result is False:
                        break
                    else:
                        new_key = tmp[:result] + (fin_key,) + tmp[result+2:]
                        tmp = new_key
                        change=True

                if change:
                    change_dict[tmp] = t_key

        # 对词频表进行更新
        for key,value in change_dict.items():
            tokens[key] = tokens.pop(value)
        del change_dict

        progress.update(1)
    end = time.time()
    print('分词器训练时间：{:.4f}s'.format(end-start))

    return vocab,merges

if __name__ == '__main__':
    input_path = 'data/TinyStoriesV2-GPT4-valid.txt'
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab,merges = tokenizer(input_path,vocab_size,special_tokens)