# Get all raw heap files and corresponding json files
import json
import networkx as nx
from tqdm.notebook import tqdm
from classes import HeapGraph, TestHeap
from time import time as timer

def get_dataset_file_paths(path, deploy=False):
    import glob
    import os
    paths = []

    file_paths = []
    key_paths = []

    sub_dir = os.walk(path)
    for directory in sub_dir:
        paths.append(directory[0])

    paths = set(paths)
    for path in paths:
        # print(os.listdir(path))
        files = glob.glob(os.path.join(path, '*.raw'), recursive=False)

        if len(files) == 0:
            continue

        for file in files:
            key_file = file[:-9] + ".json"
            if os.path.exists(key_file) and deploy is False:
                file_paths.append(file)
                key_paths.append(key_file)

            elif deploy is True:
                file_paths.append(file)

            else:
                print("Corresponding Key file does not exist for :%s" % file)

    return file_paths, key_paths
    
    
# Open the raw file and create graph structure from the pointers
def load_and_clean_heap(heap_path, json_path):
    with open(heap_path, 'rb') as fp:
        heap = bytearray(fp.read())

    with open(json_path, 'r') as fp:
        info = json.load(fp)

    # construct graph of openssh' heap
    base_addr = info.get('HEAP_START', '00000000')
    ssh_struct_addr = str(info.get('SSH_STRUCT_ADDR', None))

    if ssh_struct_addr is None or ssh_struct_addr == 'None':
        return None, None, None

    ssh_struct_addr = ssh_struct_addr.upper()
    heap_obj = HeapGraph(heap=heap, base_address=base_addr, ssh_struct_addr=ssh_struct_addr)
    heap_obj.create_graph()

    return heap_obj.heap_graph, ssh_struct_addr, info


def generate_dataset(heap_paths, json_paths, train_subset=True, block_size=100):
    limit = len(heap_paths)
    if train_subset is True:
        limit = min(block_size, limit)

    dataset = []
    labels = []

    total_files_found = 0

    for idx in tqdm(range(limit), desc='Data Files'):

        # Read the raw heap and json information and create the graph
        heap_graph, ssh_struct_addr, info = load_and_clean_heap(heap_path=heap_paths[idx],
                                                                json_path=json_paths[idx])

        if heap_graph is None:
            continue

        relevant_nodes = list((info.get('NEWKEYS_1_ADDR').upper(), info.get('NEWKEYS_2_ADDR').upper()))
        if len(relevant_nodes) == 0:
            continue

        total_files_found += 1

        # Extract state of node(size, number of outgoing edges, parent size, parent outgoing edges, offset)
        for node in list(nx.nodes(heap_graph)):
            node_info = heap_graph.nodes.get(node)
            size = node_info.get('size', 0)
            pointer_count = node_info.get('pointer_count', 0)
            pointer_offset = node_info.get('pointer_offset')
            valid_pointer_offset = node_info.get('valid_pointer_offset')

            out_degree = heap_graph.out_degree[node]

            # Create the feature vector
            feature_vector = [size, pointer_count, out_degree, pointer_offset, valid_pointer_offset]

            dataset.append(feature_vector)
            label = 0
            ## My modified code stores the addresses as numbers instead of hex encoded strings.
            ## so to compare it with relevant_nodes, we need to convert it to a hex string and strip the "0x"
            if hex(node)[2:].upper() in relevant_nodes:
                label = 1
            labels.append(label)

    # print('Total files found: %d' % total_files_found)
    return dataset, labels


# Extract actual keys
def test(heap_path, base_address, clf):
    with open(heap_path, 'rb') as fp:
        heap = bytearray(fp.read())

    obj = TestHeap(heap=heap, base_address=base_address, clf=clf)
    relevant_addresses_found = obj.test(clf=clf)

    keys_dict = dict()
    for address in relevant_addresses_found:
        data_addr = obj.resolve_pointer_address(address=address)

        # Get the allocation size of the NEW KEYS STRUCT
        allocation_size = obj.get_allocation_size(data_addr)

        # NEW_KEYS size is atleast 120 bytes
        if allocation_size < 110:
            continue
        # https://github.com/openssh/openssh-portable/blob/d9dbb5d9a0326e252d3c7bc13beb9c2434f59409/kex.h#L130
        #   struct sshenc {
        #   char	*name;
        #   const struct sshcipher *cipher;
        #   int	enabled;
        #   u_int	key_len;
        #   u_int	iv_len;
        #   u_int	block_size;
        #   u_char	*key;
        #   u_char	*iv;
        #  };
        # check if the next address is a pointer.
        # The first address is the pointer to the name
        current_key_info = dict()
        aligned_heap_addr = int(data_addr / 8)
        ## Keeping it for now,but should better be is_heap_address_valid?
        if not obj.is_pointer_candidate(obj.aligned_heap[aligned_heap_addr]):
            continue

        # resolve the pointer address after converting it to big-endian
        actual_address = obj.aligned_heap[aligned_heap_addr]
        # Transforms a pointer into an offset relative to the beginning of the heap
        actual_address = obj.resolve_pointer_address(actual_address)

        ## hr-TODO-note
        ## Heap::get_allocation_size takes as argument a pointer (and internally calls resolve_pointer_address)
        ## TestHeap::get_allocation_size takes as asrgument a relative heap offset (so you need to call resolve_pointer_address before
        ## This is not really nice code style
        ## My suggestion is to clean this up, make TestHeap inherit from Heap (and thus use the get_allocation_size implementation
        ## from Heap
        ## This requires changing the code her to pass the real pointer, not the relative address to get_allocation_size

        # Get string allocation size
        cipher_name_allocation_size = obj.get_allocation_size(actual_address)
        cipher_name = ''.join(map(chr,
                                  obj.heap[actual_address:actual_address + cipher_name_allocation_size])).strip('\x00')
        current_key_info['CIPHER'] = cipher_name

        # Get the size of the key
        key_size = int.from_bytes(obj.heap[data_addr + 20:data_addr + 24], "little")
        key_address = obj.aligned_heap[aligned_heap_addr + 4]
        actual_key_address = int(obj.resolve_pointer_address(key_address) / 8)
        num_rows = round(key_size / 8)
        ### TODO - this is probably not right as item is now a mp.uint64 instead of a byte array
        key = ''.join([hex(item)[2:] for item in obj.aligned_heap[actual_key_address:actual_key_address + num_rows]])
        current_key_info['KEY_LEN'] = key_size
        current_key_info['KEY'] = key

        iv_size = int.from_bytes(obj.heap[data_addr + 24:data_addr + 28], "little")
        if iv_size != 0:
            # iv_address = obj.convert_to_big_endian(obj.formatted_heap[aligned_heap_addr + 5])
            iv_address = obj.aligned_heap[aligned_heap_addr + 5]
            actual_iv_address = int(obj.resolve_pointer_address(iv_address) / 8)
            num_rows = round(iv_size / 8)
            ### TODO - this is probably not right as item is now a mp.uint64 instead of a byte array
            iv = ''.join([hex(item)[2:] for item in obj.aligned_heap[actual_iv_address:actual_iv_address + num_rows]])
            current_key_info['IV_LEN'] = iv_size
            current_key_info['IV'] = iv[:iv_size * 2]

        keys_dict[address] = current_key_info

    return keys_dict


def get_data_for_testing(clf, root):
    heap_paths, json_paths = get_dataset_file_paths(root)
    both_keys = 0
    one_key = 0
    zero_keys = 0
    total_files = 0

    key_list = []
    found_key_count = 0
    total_individual_keys = 0

    heaps_without_newkey_addr = []
    no_key_found = []
    one_key_found = []

    for idx in range(len(json_paths)):
        print(idx)
        with open(json_paths[idx], 'r') as fp:
            info = json.load(fp)
            base_address = info.get('HEAP_START', None)
            if base_address is None:
                continue
            key_dict = test(heap_path=heap_paths[idx], base_address=base_address, clf=clf)

            newkeys_1 = info.get('NEWKEYS_1_ADDR', None)
            newkeys_2 = info.get('NEWKEYS_2_ADDR', None)

            if newkeys_1 is None or newkeys_2 is None:
                heaps_without_newkey_addr.append(json_paths[idx] + '\n')
                key_list.append(info.get('KEY_A', '').upper())
                key_list.append(info.get('KEY_B', '').upper())
                key_list.append(info.get('KEY_C', '').upper())
                key_list.append(info.get('KEY_D', '').upper())
                key_list.append(info.get('KEY_E', '').upper())
                key_list.append(info.get('KEY_F', '').upper())

                # Remove empty keys
                key_list = [x for x in key_list if x != '']
                total_individual_keys += min(len(key_list), 2)

                for found, key_struct in key_dict.items():
                    found_key = key_struct.get('KEY', None)
                    if found_key.upper() in key_list:
                        found_key_count += 1

                    if key_struct.get('IV_LEN', 0) > 0:
                        found_iv = key_struct.get('IV', None)
                        if found_iv.upper() in key_list:
                            found_key_count += 1

            else:
                found_new_keys1 = key_dict.get(newkeys_1.upper(), None)
                found_new_keys2 = key_dict.get(newkeys_2.upper(), None)

                if found_new_keys1 is not None and found_new_keys2 is not None:
                    # print('Found both new keys')
                    both_keys += 1

                elif found_new_keys2 is not None or found_new_keys1 is not None:
                    # print('Found one new key')
                    one_key_found.append(json_paths[idx] + '\n')
                    one_key += 1

                else:
                    # print('Found no keys')
                    no_key_found.append(json_paths[idx] + '\n')
                    zero_keys += 1

                total_files += 1
    return both_keys, one_key, zero_keys, total_files, found_key_count, total_individual_keys, one_key_found, \
        no_key_found

