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

class VocabHeap:
    def __init__(self,vo):
        pass

def tokenizer(input_path,vocab_size,special_tokens):
    start  = time.time()
    num_processes = os.cpu_count() * 2
    # special_token = 
    ## Usage

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, special_tokens)
            
    # The following is a serial implementation, but you can parallelize this 
    # by sending each start/end pair to a set of processes.
    tokens = {}
    tasks = zip(boundaries[:-1], boundaries[1:])


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

    tokens_rep = {}
    pair_loc = {}
    for key,value in tokens.items():
        tokens_rep[key] = key

    vocab = {i:tuple(chr(i).encode('utf-8')) for i in range(256)}
    merges = [tuple(chr(i).encode('utf-8')) for i in range(256)]

    merges.append(tuple(special_token.encode('utf-8')))
    vocab[tuple(special_token.encode('utf-8'))] = len(merges)-1
    
    start = time.time()
    progress = tqdm(total=vocab_size-len(vocab), desc="分词器训练进度")
    
    # 进行第一次更新
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
            # 确定元组位置
            pair_set = pair_loc.get(e_f+e_b,set())
            pair_set.add(t_key)
            pair_loc[e_f+e_b] = pair_set

    search_time = 0
    update_time = 0
    while(len(vocab)<vocab_size):
        # 获取词频最高的token
        start_s = time.time()
        max_value = max(tmp_frequncy.values())
        max_keys = [k for k, v in tmp_frequncy.items() if v == max_value]
        fin_key = max(max_keys)
        #加入到词汇表里
        merges.append(fin_key)
        vocab[fin_key] = len(merges)-1
        end_s = time.time()
        search_time = search_time + (end_s-start_s)
        start_u = time.time()
        # 对词频表进行更新
        change_dict = {}
        # t_key是原始形态，方便查找词组位置
        # tokens_rep[t_key]是合并后的形态
        for t_key in pair_loc[fin_key]:
            if t_key in merges:
                continue
            else:
                value = tokens[t_key]
                tmp = tokens_rep[t_key]
                while(True):
                    # 判断当前键里有没有最新的高频token
                    result = detect(fin_key,tmp)
                    if result is False:
                        break
                    else:
                        tmp_len = len(tmp)
                        if tmp_len > 2:
                            # 需要合并的两个token在开头，且长度大于等于3,则合并token后，后面跟着的那个token对的出现数量要降低,新出现的token数量增加
                            if result == 0:
                                e_0 = convert(tmp[0])
                                e_f = convert(tmp[1])
                                e_b = convert(tmp[2])
                                tmp_frequncy[e_f+e_b] = tmp_frequncy[e_f+e_b] - value
                                if tmp_frequncy[e_f+e_b] ==0 :
                                    tmp_frequncy.pop(e_f+e_b)
                                    pair_loc.pop(e_f+e_b)
                                tmp_frequncy[e_0+e_f+e_b] = tmp_frequncy.get(e_0+e_f+e_b,0) + value
                                pair_set = pair_loc.get(e_0+e_f+e_b,set())
                                pair_set.add(t_key)
                                pair_loc[e_0+e_f+e_b] = pair_set

                            # 需要合并的两个token在结尾，且长度大于等于3,则合并token后，前面跟着的那个token对的出现数量要降低，新出现的token数量增加
                            elif result == tmp_len-2:
                                e_f = convert(tmp[-3])
                                e_b = convert(tmp[-2])
                                e_fin = convert(tmp[-1])
                                tmp_frequncy[e_f+e_b] = tmp_frequncy[e_f+e_b] - value
                                if tmp_frequncy[e_f+e_b] ==0 :
                                    tmp_frequncy.pop(e_f+e_b)
                                    pair_loc.pop(e_f+e_b)
                                tmp_frequncy[e_f+e_b+e_fin] = tmp_frequncy.get(e_f+e_b+e_fin,0) + value

                                pair_set = pair_loc.get(e_f+e_b+e_fin,set())
                                pair_set.add(t_key)
                                pair_loc[e_f+e_b+e_fin] = pair_set
                                
                            # 需要合并的两个token在中间，且长度大于等于4,则合并token后，前后跟着的那个token对的出现数量要降低，新出现的token数量增加
                            else:
                                e_ff = convert(tmp[result-1])
                                e_f = convert(tmp[result])
                                e_b = convert(tmp[result+1])
                                e_bb = convert(tmp[result+2])
                                tmp_frequncy[e_ff+e_f] = tmp_frequncy[e_ff+e_f] - value  
                                if tmp_frequncy[e_ff+e_f] == 0:
                                    tmp_frequncy.pop(e_ff+e_f)
                                    pair_loc.pop(e_ff+e_f)
                                tmp_frequncy[e_b+e_bb] = tmp_frequncy[e_b+e_bb] - value
                                if tmp_frequncy[e_b+e_bb] ==0:
                                    tmp_frequncy.pop(e_b+e_bb)
                                    pair_loc.pop(e_b+e_bb)
                                tmp_frequncy[e_ff+e_f+e_b] = tmp_frequncy.get(e_ff+e_f+e_b,0) + value
                                tmp_frequncy[e_f+e_b+e_bb] = tmp_frequncy.get(e_f+e_b+e_bb,0) + value
                                pair_set = pair_loc.get(e_ff+e_f+e_b,set())
                                pair_set.add(t_key)
                                pair_loc[e_ff+e_f+e_b] = pair_set
                                pair_set = pair_loc.get(e_f+e_b+e_bb,set())
                                pair_set.add(t_key)
                                pair_loc[e_f+e_b+e_bb] = pair_set

                        new_key = tmp[:result] + (fin_key,) + tmp[result+2:]
                        tmp = new_key
                        
                change_dict[t_key] = tmp

        # 对词频表进行更新
        for key,value in change_dict.items():
            tokens_rep[key] = value

        del change_dict
        # 已经合并新分词了，将该分词从原来的词频表中去掉
        tmp_frequncy.pop(fin_key)
        end_u = time.time()
        update_time = update_time + end_u-start_u

        progress.update(1)
        progress.set_postfix({'词频表长度':len(tmp_frequncy),'分词表长度':len(tokens)})

    end = time.time()
    progress.close()
    print('分词器训练时间：{:.4f}s'.format(end-start))
    print('总搜索时间: {:.4f}s'.format(search_time))
    print('总更新时间: {:.4f}s'.format(update_time))

    return vocab,merges

if __name__ == '__main__':
    input_path = 'data/TinyStoriesV2-GPT4-valid.txt'
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab,merges = tokenizer(input_path,vocab_size,special_tokens)
    print(len(merges))