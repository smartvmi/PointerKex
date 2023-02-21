import re
import networkx as nx


class Heap:

    base_address = None
    pointer_list = None
    heap = None
    ssh_struct_address = None
    aligned_heap = None
    formatted_heap = None
    length = 0
    heap_size = 0
    pointer_offsets = None
    regex = re.compile("[0-9a-fA-F]{11}[1-9a-fA-F]0{4}")

    def __init__(self, base_address, heap, ssh_struct_addr=None):
        import numpy as np
        self.base_address = base_address
        self.ssh_struct_address = ssh_struct_addr.lstrip("0").upper()

        self.heap = heap
        self.heap_size = len(heap)
        self.length = int(self.heap_size/8)
        self.aligned_heap = np.reshape(heap,  newshape=(self.length, 8))
        self.pointer_list = None
        self.format_heap(heap=heap)

        self.get_all_pointers(ssh_struct_addr=ssh_struct_addr)

    def is_pointer(self, address):
        if self.regex.match(address) is not None:
            return True

        return False

    def format_heap(self, heap):
        idx = 0
        self.formatted_heap = []
        formatted_heap = ''.join(format(x, '02x') for x in heap).upper()
        # Each byte is two characters
        while idx < len(formatted_heap):
            self.formatted_heap.append(formatted_heap[idx:idx + 16])
            idx += 16

    def is_heap_address_valid(self, address):
        if int(address, 16) <= int(self.base_address, 16) or \
                int(address, 16) > (int(self.base_address, 16) + self.heap_size):
            return False

        return True

    def resolve_pointer_address(self, address):
        """
        Gets the actual offset from the starting of the heap
        :param address: Valid address in the heap
        :return: Actual offset of the heap in base10
        """
        diff = int(address, 16) - int(self.base_address, 16)
        if diff <= 0 or diff > self.heap_size:
            return 0
        return diff

    def get_all_pointers(self, ssh_struct_addr=None):
        # We do the rstrip here because it is little endian
        ptr = [(self.formatted_heap[x], x) for x in range(self.length)
               if self.regex.match(self.formatted_heap[x]) is not None]

        self.pointer_list = [x[0] for x in ptr]
        self.pointer_offsets = [x[1] for x in ptr]
        return

    def get_raw_size(self):
        return self.length

    def get_aligned_size(self):
        return len(self.aligned_heap)

    def get_allocation_size(self, pointer):
        if self.is_heap_address_valid(pointer) is False:
            return 0

        # Get the heap offset by resolving the pointer
        heap_offset = self.resolve_pointer_address(pointer)

        # Get the 8-byte aligned offset for formatted heap
        heap_offset = int(heap_offset/8)

        # Header info is in the previous byte in little endian form
        header_offset = heap_offset - 1
        header_data = self.formatted_heap[header_offset]

        if int(self.convert_to_big_endian(header_data), 16) <= 0:
            return 0

        # 8 bytes for the malloc header, 1 byte for flags
        size = int(self.convert_to_big_endian(header_data), 16) - 9

        if size > self.heap_size:
            return 0
        return size

    def get_data_at_index(self, index):
        return self.formatted_heap[index]

    @staticmethod
    def convert_to_big_endian(data):
        """
        Function converts hex little endian to big endian
        :param data:
        :return:
        """
        temp = bytearray.fromhex(data)
        temp.reverse()
        return ''.join(format(x, '02x') for x in temp).upper()
        

class HeapGraph(Heap):
    heap_graph = None
    starting_points = []

    def __init__(self, base_address, heap, ssh_struct_addr=None):
        self.heap_graph = None
        super().__init__(base_address=base_address, heap=heap, ssh_struct_addr=ssh_struct_addr)

    def get_starting_points(self):
        from copy import deepcopy
        return deepcopy(self.starting_points)

    def get_graph(self):
        return self.heap_graph

    def create_graph(self):

        self.heap_graph = nx.DiGraph()

        # Big endian valid pointers
        pointers = [self.convert_to_big_endian(pointer) for pointer in self.pointer_list
                    if self.is_heap_address_valid(self.convert_to_big_endian(pointer))]

        # Select only pointers which have addresses within the heap
        pointers = [x.lstrip("0") for x in pointers if self.is_heap_address_valid(x) is True]

        if self.ssh_struct_address is not None:
            pointers.append(self.ssh_struct_address)

        # Take only unique pointers. Multiple unique addresses are not necessary
        pointers = set(pointers)

        #pointer that points to a data structure should be 16 bytes align (malloc in 64bit is 16 bytes allign)
        #https://www.gnu.org/software/libc/manual/html_node/Aligned-Memory-Blocks.html
        pointers = [x for x in pointers if int(x, 16) % 16 == 0]
        # print(pointers)

        # Create all the nodes
        for pointer in pointers:
            allocated_size = self.get_allocation_size(pointer=pointer)
            if allocated_size <= 0:
                continue

            num_pointers = self.count_pointers(pointer)
            self.heap_graph.add_node(pointer.lstrip("0"), size=allocated_size,
                                     pointer_count=num_pointers, offset=0)

        # Start adding edges
        for idx, pointer in enumerate(pointers):

            # Find how many rows are contained in the heap
            struct_start_addr = int(self.resolve_pointer_address(address=pointer) / 8)
            # Get the search boundary from the allocated size stored in the node
            allocated_size = self.heap_graph.nodes.get(pointer, {}).get('size', 0)
            if allocated_size == 0:
                continue
            struct_ending_addr = struct_start_addr + int(allocated_size / 8)

            inner_idx = struct_start_addr
            while inner_idx <= struct_ending_addr:
                if self.is_pointer(self.formatted_heap[inner_idx]) is True:
                    # We found a pointer, we add it as an edge from pointer to the identified address
                    # Convert the v edge to big endian and set the offset as the edge property
                    # Remove self loops by checking whether both the pointers are identical
                    target_pointer = self.convert_to_big_endian(self.formatted_heap[inner_idx]).lstrip("0")
                    if self.heap_graph.has_node(pointer) and self.heap_graph.has_node(target_pointer) and \
                            pointer != target_pointer:

                        #set offset on the node as well
                        self.heap_graph.nodes.get(target_pointer, {}).update(
                            {"offset": (inner_idx - struct_start_addr) * 8})

                        self.heap_graph.add_edge(u_of_edge=pointer,
                                                 v_of_edge=target_pointer,
                                                 offset=(inner_idx - struct_start_addr) * 8)
                inner_idx += 1

        # remove nodes that do not have any incoming or outgoing edges
        self.heap_graph.remove_nodes_from(list(nx.isolates(self.heap_graph)))

        # Remove isolated pairs of nodes
        starting_points = [x for x in self.heap_graph.nodes if self.heap_graph.in_degree(x) == 0]
        nodes_to_be_removed = []
        for node in starting_points:
            if self.heap_graph.out_degree(node) == 1 and \
                    len(list(self.heap_graph.neighbors(list(self.heap_graph.neighbors(node))[0]))) == 0:
                nodes_to_be_removed.append(node)

        # Remove the nodes
        self.heap_graph.remove_nodes_from(nodes_to_be_removed)
        # Remove the neighbors of the deleted nodes which are now isolated nodes
        self.heap_graph.remove_nodes_from(list(nx.isolates(self.heap_graph)))

        # Update the starting points
        self.starting_points = [x for x in self.heap_graph.nodes if self.heap_graph.in_degree(x) == 0]

    def count_pointers(self, node):
        # The number of pointers for the allocated memory
        num_pointers = 0
        allocation_size = int(self.get_allocation_size(node) / 8)
        starting_addr = int(self.resolve_pointer_address(node) / 8)
        idx = 0
        while idx < allocation_size:
            if self.is_pointer(self.formatted_heap[starting_addr + idx]) is True:
                num_pointers += 1
            idx += 1
        return num_pointers
        
        
class TestHeap:

    base_address = None
    heap = None
    aligned_heap = None
    formatted_heap = None
    length = 0
    heap_size = 0
    regex = re.compile("[0-9a-fA-F]{11}[1-9a-fA-F]0{4}")

    def __init__(self, base_address, heap, clf):
        import numpy as np
        self.base_address = base_address

        self.heap = heap
        self.heap_size = len(heap)
        self.length = int(self.heap_size/8)
        self.aligned_heap = np.reshape(heap,  newshape=(self.length, 8))
        self.pointer_list = None
        self.clf = clf
        self.format_heap(heap)

    def is_pointer(self, address):
        if self.regex.match(address) is not None:
            return True

        return False

    def format_heap(self, heap):
        idx = 0
        self.formatted_heap = []
        formatted_heap = ''.join(format(x, '02x') for x in heap).upper()
        # Each byte is two characters
        while idx < len(formatted_heap):
            self.formatted_heap.append(formatted_heap[idx:idx + 16])
            idx += 16

    def is_heap_address_valid(self, idx):
        if idx > len(self.aligned_heap):
            return False

        address = ''.join(format(x, '02x') for x in self.aligned_heap[idx][::-1]).upper()
        if int(address, 16) <= int(self.base_address, 16) or \
                int(address, 16) > (int(self.base_address, 16) + self.heap_size):
            return False

        return True

    def resolve_pointer_address(self, address):
        """
        Gets the actual offset from the starting of the heap
        :param address: Valid address in the heap
        :return: Actual offset of the heap in base10
        """
        diff = int(address, 16) - int(self.base_address, 16)
        if diff <= 0 or diff > self.heap_size:
            return 0
        return diff

    def get_raw_size(self):
        return self.length

    def get_aligned_size(self):
        return len(self.aligned_heap)

    def get_allocation_size(self, heap_offset):

        # Get the 8-byte aligned offset for formatted heap
        heap_offset = int(heap_offset/8)

        # Header info is in the previous byte in little endian form
        header_offset = heap_offset - 1
        header_data = self.formatted_heap[header_offset]

        size = int(self.convert_to_big_endian(header_data), 16)
        if size <= 0:
            return 0

        # 8 bytes for the malloc header, 1 byte for flags
        size = size - 9

        if size > self.heap_size:
            return 0
        return size

    def get_data_at_index(self, index):
        return self.formatted_heap[index]

    @staticmethod
    def convert_to_big_endian(data):
        """
        Function converts hex little endian to big endian
        :param data:
        :return:
        """
        temp = bytearray.fromhex(data)
        temp.reverse()
        return ''.join(format(x, '02x') for x in temp).upper()

    def test(self, clf):

        graph = nx.DiGraph()

        relevant_addresses = []
        block_pointers = []
        address_block = []
        size = self.get_aligned_size()
        for idx in range(size):
            curr_row = self.formatted_heap[idx]
            if self.is_pointer(curr_row) and self.is_heap_address_valid(idx=idx):
                address = ''.join(format(x, '02x') for x in self.aligned_heap[idx][::-1]).upper()
                data_addr = self.resolve_pointer_address(address=address)
                if data_addr == 0:
                    continue
                # [size, pointer_count, offset, out_degree]
                # Currently, offset is not included as it requires information about the predecessor

                size = self.get_allocation_size(heap_offset=data_addr)
                data_addr = int(data_addr/8)
                indices_to_check = int(size/8)
                out_degree = 0
                pointer_count = 0

                # Get the heap offset by resolving the pointer
                num_pointers = self.count_pointers(starting_addr=data_addr, allocation_size=int(size/8))
                for idx_range in range(indices_to_check):
                    if self.is_pointer(self.formatted_heap[data_addr+idx_range]) is True:
                        pointer_count += 1

                        if self.is_heap_address_valid(data_addr+idx_range) is True:
                            out_degree += 1
                block_pointers.append([size, pointer_count, out_degree])
                address_block.append(address.lstrip('0'))
                # print([size, pointer_count, out_degree])
            if len(block_pointers) >= 100:
                y_pred = clf.predict(block_pointers)

                for ptr_idx in range(len(y_pred)):
                    if y_pred[ptr_idx] == 1:
                        #  if y_pred == 1:
                        relevant_addresses.append(address_block[ptr_idx])

                block_pointers = []
                address_block = []

        if len(block_pointers) > 0:
            y_pred = clf.predict(block_pointers)
            for ptr_idx in range(len(y_pred)):
                if y_pred[ptr_idx] == 1:
                    #  if y_pred == 1:
                    relevant_addresses.append(address_block[ptr_idx])
        return list(set(relevant_addresses))

    def count_pointers(self, starting_addr, allocation_size):
        # The number of pointers for the allocated memory
        num_pointers = 0
        idx = 0
        while idx < allocation_size:
            if self.is_pointer(self.formatted_heap[starting_addr + idx]) is True:
                num_pointers += 1
            idx += 1
        return num_pointers
