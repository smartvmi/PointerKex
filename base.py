# This file contains the class for the key extraction notebook
import re


class Heap:
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
        self.base_address_int = int(base_address, 16)
        self.heap = heap
        self.heap_size = len(heap)
        self.end_of_heap = self.base_address_int + self.heap_size
        self.length = int(self.heap_size / 8)
        self.aligned_heap = np.reshape(heap, newshape=(self.length, 8))
        self.aligned_heap_size = len(self.aligned_heap)
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
        # formatted_heap = ''.join(format(x, '02x') for x in heap).upper()
        formatted_heap = ''.join(hex_dict[x] for x in heap)

        # Each byte is two characters
        self.formatted_heap = [formatted_heap[i:i + 16] for i in range(0, len(formatted_heap), 16)]
        # while idx < len(formatted_heap):
        #     self.formatted_heap.append(formatted_heap[idx:idx + 16])
        #     idx += 16

    def is_heap_address_valid(self, idx):
        if idx > self.aligned_heap_size:
            return False

        # address = ''.join(format(x, '02x') for x in self.aligned_heap[idx][::-1]).upper()
        address = int(self.formatted_heap[idx][-2] + self.formatted_heap[idx][-1] + self.formatted_heap[idx][-4] + \
                      self.formatted_heap[idx][-3] + self.formatted_heap[idx][-6] + self.formatted_heap[idx][-5] + \
                      self.formatted_heap[idx][-8] + self.formatted_heap[idx][-7] + self.formatted_heap[idx][-10] + \
                      self.formatted_heap[idx][-9] + self.formatted_heap[idx][-12] + self.formatted_heap[idx][-11] + \
                      self.formatted_heap[idx][-14] + self.formatted_heap[idx][-13] + self.formatted_heap[idx][-16] + \
                      self.formatted_heap[idx][-15], 16)

        if address <= self.base_address_int or address > self.end_of_heap:
            return False

        return True

    def resolve_pointer_address(self, address):
        """
        Gets the actual offset from the starting of the heap
        :param address: Valid address in the heap
        :return: Actual offset of the heap in base10
        """
        diff = int(address, 16) - self.base_address_int
        if diff <= 0 or diff > self.heap_size:
            return 0
        return diff

    def get_raw_size(self):
        return self.length

    @property
    def aligned_size(self):
        return self.aligned_heap_size

    def get_allocation_size(self, heap_offset):

        # Get the 8-byte aligned offset for formatted heap
        heap_offset = int(heap_offset / 8)

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
        # temp = bytearray.fromhex(data)
        # temp.reverse()
        # return ''.join(format(x, '02x') for x in temp).upper()
        return data[-2] + data[-1] + data[-4] + data[-3] + data[-6] + data[-5] + data[-8] + data[-7] + data[-10] + \
            data[-9] + data[-12] + data[-11] + data[-14] + data[-13] + data[-16] + data[-15]

    def test(self, clf):

        relevant_addresses = []
        block_pointers = []
        address_block = []
        heap_size = self.aligned_size
        newkeys_found = 0
        for idx in range(heap_size):
            curr_row = self.formatted_heap[idx]
            if self.is_pointer(curr_row) and self.is_heap_address_valid(idx=idx):
                # address = ''.join(format(x, '02x') for x in self.aligned_heap[idx][::-1]).upper()
                address = self.formatted_heap[idx][-2] + self.formatted_heap[idx][-1] + self.formatted_heap[idx][-4] + \
                          self.formatted_heap[idx][-3] + self.formatted_heap[idx][-6] + self.formatted_heap[idx][-5] + \
                          self.formatted_heap[idx][-8] + self.formatted_heap[idx][-7] + self.formatted_heap[idx][-10] + \
                          self.formatted_heap[idx][-9] + self.formatted_heap[idx][-12] + \
                          self.formatted_heap[idx][-11] + self.formatted_heap[idx][-14] + \
                          self.formatted_heap[idx][-13] + self.formatted_heap[idx][-16] + self.formatted_heap[idx][-15]
                data_addr = self.resolve_pointer_address(address=address)
                if data_addr == 0:
                    continue
                # [size, pointer_count, offset, out_degree]
                # Currently, offset is not included as it requires information about the predecessor

                size = self.get_allocation_size(heap_offset=data_addr)
                data_addr = int(data_addr / 8)
                indices_to_check = int(size / 8) + 1
                out_degree = 0
                pointer_count = 0

                # Get the heap offset by resolving the pointer
                # num_pointers = self.count_pointers(starting_addr=data_addr, allocation_size=int(size / 8))
                for idx_range in range(indices_to_check):
                    if self.is_pointer(self.formatted_heap[data_addr + idx_range]) is True:
                        pointer_count += 1
                        final_pointer_offset = idx_range
                        if self.is_heap_address_valid(data_addr + idx_range) is True:
                            out_degree += 1
                            final_valid_pointer_offset = idx_range
                block_pointers.append([size, pointer_count, out_degree, final_pointer_offset,
                                       final_valid_pointer_offset])
                address_block.append(address.lstrip('0'))
                # print([size, pointer_count, out_degree])
            if len(block_pointers) >= 250:
                y_pred = clf.predict(block_pointers)

                for ptr_idx in range(len(y_pred)):
                    if y_pred[ptr_idx] == 1:
                        #  if y_pred == 1:
                        relevant_addresses.append(address_block[ptr_idx])

                block_pointers = []
                address_block = []
                # newkeys_found += sum(y_pred)
                # if newkeys_found >= 2:
                #     break

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


hex_dict = dict()
hex_dict[0] = '00'
hex_dict[1] = '01'
hex_dict[2] = '02'
hex_dict[3] = '03'
hex_dict[4] = '04'
hex_dict[5] = '05'
hex_dict[6] = '06'
hex_dict[7] = '07'
hex_dict[8] = '08'
hex_dict[9] = '09'
hex_dict[10] = '0A'
hex_dict[11] = '0B'
hex_dict[12] = '0C'
hex_dict[13] = '0D'
hex_dict[14] = '0E'
hex_dict[15] = '0F'
hex_dict[16] = '10'
hex_dict[17] = '11'
hex_dict[18] = '12'
hex_dict[19] = '13'
hex_dict[20] = '14'
hex_dict[21] = '15'
hex_dict[22] = '16'
hex_dict[23] = '17'
hex_dict[24] = '18'
hex_dict[25] = '19'
hex_dict[26] = '1A'
hex_dict[27] = '1B'
hex_dict[28] = '1C'
hex_dict[29] = '1D'
hex_dict[30] = '1E'
hex_dict[31] = '1F'
hex_dict[32] = '20'
hex_dict[33] = '21'
hex_dict[34] = '22'
hex_dict[35] = '23'
hex_dict[36] = '24'
hex_dict[37] = '25'
hex_dict[38] = '26'
hex_dict[39] = '27'
hex_dict[40] = '28'
hex_dict[41] = '29'
hex_dict[42] = '2A'
hex_dict[43] = '2B'
hex_dict[44] = '2C'
hex_dict[45] = '2D'
hex_dict[46] = '2E'
hex_dict[47] = '2F'
hex_dict[48] = '30'
hex_dict[49] = '31'
hex_dict[50] = '32'
hex_dict[51] = '33'
hex_dict[52] = '34'
hex_dict[53] = '35'
hex_dict[54] = '36'
hex_dict[55] = '37'
hex_dict[56] = '38'
hex_dict[57] = '39'
hex_dict[58] = '3A'
hex_dict[59] = '3B'
hex_dict[60] = '3C'
hex_dict[61] = '3D'
hex_dict[62] = '3E'
hex_dict[63] = '3F'
hex_dict[64] = '40'
hex_dict[65] = '41'
hex_dict[66] = '42'
hex_dict[67] = '43'
hex_dict[68] = '44'
hex_dict[69] = '45'
hex_dict[70] = '46'
hex_dict[71] = '47'
hex_dict[72] = '48'
hex_dict[73] = '49'
hex_dict[74] = '4A'
hex_dict[75] = '4B'
hex_dict[76] = '4C'
hex_dict[77] = '4D'
hex_dict[78] = '4E'
hex_dict[79] = '4F'
hex_dict[80] = '50'
hex_dict[81] = '51'
hex_dict[82] = '52'
hex_dict[83] = '53'
hex_dict[84] = '54'
hex_dict[85] = '55'
hex_dict[86] = '56'
hex_dict[87] = '57'
hex_dict[88] = '58'
hex_dict[89] = '59'
hex_dict[90] = '5A'
hex_dict[91] = '5B'
hex_dict[92] = '5C'
hex_dict[93] = '5D'
hex_dict[94] = '5E'
hex_dict[95] = '5F'
hex_dict[96] = '60'
hex_dict[97] = '61'
hex_dict[98] = '62'
hex_dict[99] = '63'
hex_dict[100] = '64'
hex_dict[101] = '65'
hex_dict[102] = '66'
hex_dict[103] = '67'
hex_dict[104] = '68'
hex_dict[105] = '69'
hex_dict[106] = '6A'
hex_dict[107] = '6B'
hex_dict[108] = '6C'
hex_dict[109] = '6D'
hex_dict[110] = '6E'
hex_dict[111] = '6F'
hex_dict[112] = '70'
hex_dict[113] = '71'
hex_dict[114] = '72'
hex_dict[115] = '73'
hex_dict[116] = '74'
hex_dict[117] = '75'
hex_dict[118] = '76'
hex_dict[119] = '77'
hex_dict[120] = '78'
hex_dict[121] = '79'
hex_dict[122] = '7A'
hex_dict[123] = '7B'
hex_dict[124] = '7C'
hex_dict[125] = '7D'
hex_dict[126] = '7E'
hex_dict[127] = '7F'
hex_dict[128] = '80'
hex_dict[129] = '81'
hex_dict[130] = '82'
hex_dict[131] = '83'
hex_dict[132] = '84'
hex_dict[133] = '85'
hex_dict[134] = '86'
hex_dict[135] = '87'
hex_dict[136] = '88'
hex_dict[137] = '89'
hex_dict[138] = '8A'
hex_dict[139] = '8B'
hex_dict[140] = '8C'
hex_dict[141] = '8D'
hex_dict[142] = '8E'
hex_dict[143] = '8F'
hex_dict[144] = '90'
hex_dict[145] = '91'
hex_dict[146] = '92'
hex_dict[147] = '93'
hex_dict[148] = '94'
hex_dict[149] = '95'
hex_dict[150] = '96'
hex_dict[151] = '97'
hex_dict[152] = '98'
hex_dict[153] = '99'
hex_dict[154] = '9A'
hex_dict[155] = '9B'
hex_dict[156] = '9C'
hex_dict[157] = '9D'
hex_dict[158] = '9E'
hex_dict[159] = '9F'
hex_dict[160] = 'A0'
hex_dict[161] = 'A1'
hex_dict[162] = 'A2'
hex_dict[163] = 'A3'
hex_dict[164] = 'A4'
hex_dict[165] = 'A5'
hex_dict[166] = 'A6'
hex_dict[167] = 'A7'
hex_dict[168] = 'A8'
hex_dict[169] = 'A9'
hex_dict[170] = 'AA'
hex_dict[171] = 'AB'
hex_dict[172] = 'AC'
hex_dict[173] = 'AD'
hex_dict[174] = 'AE'
hex_dict[175] = 'AF'
hex_dict[176] = 'B0'
hex_dict[177] = 'B1'
hex_dict[178] = 'B2'
hex_dict[179] = 'B3'
hex_dict[180] = 'B4'
hex_dict[181] = 'B5'
hex_dict[182] = 'B6'
hex_dict[183] = 'B7'
hex_dict[184] = 'B8'
hex_dict[185] = 'B9'
hex_dict[186] = 'BA'
hex_dict[187] = 'BB'
hex_dict[188] = 'BC'
hex_dict[189] = 'BD'
hex_dict[190] = 'BE'
hex_dict[191] = 'BF'
hex_dict[192] = 'C0'
hex_dict[193] = 'C1'
hex_dict[194] = 'C2'
hex_dict[195] = 'C3'
hex_dict[196] = 'C4'
hex_dict[197] = 'C5'
hex_dict[198] = 'C6'
hex_dict[199] = 'C7'
hex_dict[200] = 'C8'
hex_dict[201] = 'C9'
hex_dict[202] = 'CA'
hex_dict[203] = 'CB'
hex_dict[204] = 'CC'
hex_dict[205] = 'CD'
hex_dict[206] = 'CE'
hex_dict[207] = 'CF'
hex_dict[208] = 'D0'
hex_dict[209] = 'D1'
hex_dict[210] = 'D2'
hex_dict[211] = 'D3'
hex_dict[212] = 'D4'
hex_dict[213] = 'D5'
hex_dict[214] = 'D6'
hex_dict[215] = 'D7'
hex_dict[216] = 'D8'
hex_dict[217] = 'D9'
hex_dict[218] = 'DA'
hex_dict[219] = 'DB'
hex_dict[220] = 'DC'
hex_dict[221] = 'DD'
hex_dict[222] = 'DE'
hex_dict[223] = 'DF'
hex_dict[224] = 'E0'
hex_dict[225] = 'E1'
hex_dict[226] = 'E2'
hex_dict[227] = 'E3'
hex_dict[228] = 'E4'
hex_dict[229] = 'E5'
hex_dict[230] = 'E6'
hex_dict[231] = 'E7'
hex_dict[232] = 'E8'
hex_dict[233] = 'E9'
hex_dict[234] = 'EA'
hex_dict[235] = 'EB'
hex_dict[236] = 'EC'
hex_dict[237] = 'ED'
hex_dict[238] = 'EE'
hex_dict[239] = 'EF'
hex_dict[240] = 'F0'
hex_dict[241] = 'F1'
hex_dict[242] = 'F2'
hex_dict[243] = 'F3'
hex_dict[244] = 'F4'
hex_dict[245] = 'F5'
hex_dict[246] = 'F6'
hex_dict[247] = 'F7'
hex_dict[248] = 'F8'
hex_dict[249] = 'F9'
hex_dict[250] = 'FA'
hex_dict[251] = 'FB'
hex_dict[252] = 'FC'
hex_dict[253] = 'FD'
hex_dict[254] = 'FE'
hex_dict[255] = 'FF'