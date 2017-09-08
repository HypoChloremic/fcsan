from __future__ import print_function
from __future__ import absolute_import

import sys, warnings, string, os
import numpy

try:
    import pandas
    pandas_found = True
except ImportError:
    print('pandas is not installed, so the parse_fcs function can only be used together with numpy.')
    pandas_found = False
except Exception as e:
    print('pandas installation is improperly configured. It raised the following error {0}'.format(e))
    pandas_found = False



class Reader:
    def __init__(self):
        self.annotation = {}



    def read_header(self, file_handle):
        """
        Reads the header of the FCS file.
        The header specifies where the annotation, data and analysis are located inside the binary file.
        """
        header = {}
        header['FCS format'] = file_handle.read(6)

        file_handle.read(4) # 4 space characters after the FCS format

        for field in ['text start', 'text end', 'data start', 'data end', 'analysis start', 'analysis end']:
            s = file_handle.read(8)
            try:
                ival = int(s)
            except ValueError as e:
                ival = 0
            header[field] = ival


        self._data_start = header['data start']
        self._data_end = header['data start']

        if header['analysis start'] != 0:
            warnings.warn('There appears to be some information in the ANALYSIS segment of file {0}. However, it might not be read correctly.'.format(self.path))

        self.annotation['__header__'] = header

    def read_text(self, file_handle):
        """
        Reads the TEXT segment of the FCS file.
        This is the meta data associated with the FCS file.
        Converting all meta keywords to lower case.
        """
        header = self.annotation['__header__'] # For convenience

        #####
        # Read in the TEXT segment of the FCS file
        # There are some differences in how the 
        file_handle.seek(header['text start'], 0)
        raw_text = file_handle.read(header['text end'] - header['text start'] + 1)

        # Parse the TEXT segment of the FCS file into a python dictionary
        delimiter = raw_text[:1]
        print(delimiter)

        # This was changed because the index [-1] gives an integer, and not the actual byte. 
        if raw_text[-1:] != delimiter:
            raw_text = raw_text.strip()
            if raw_text[-1:] != delimiter:
                raise_parser_feature_not_implemented('Parser expects the same delimiter character in beginning and end of TEXT segment')

        temp_text = raw_text[1:-1]
        raw_text_segments = temp_text.split(delimiter) # Using 1:-1 to remove first and last characters which should be reserved for delimiter


        keys, values = raw_text_segments[0::2], raw_text_segments[1::2]
        text = {key : value for key, value in zip(keys, values)} # Build dictionary

        ####
        # Extract channel names and convert some of the channel properties and other fields into numeric data types (from string)
        # Note: do not use regular expressions for manipulations here. Regular expressions are too heavy in terms of computation time.
        # Note: that we previously had a utf8 encoded text segment "$PAR", but was changed to bytearray. 
        pars = int(text[b'$PAR'])

        if b'$P0B' in keys: # Checking whether channel number count starts from 0 or from 1
            self.channel_numbers = range(0, pars) # Channel number count starts from 0
        else:
            self.channel_numbers = range(1, pars + 1) # Channel numbers start from 1

        ## Extract parameter names
        try:
            names_n = tuple([text[bytes(f'$P{i}N', "utf8")] for i in self.channel_numbers])

        except KeyError:
            names_n = []

        try:
            names_s = tuple([text[bytes(f'$P{i}S', "utf8")] for i in self.channel_numbers])
        except KeyError:
            names_s = []

        self.channel_names_s = names_s
        self.channel_names_n = names_n

        # Convert some of the fields into integer values
        keys_encoding_bits  = [bytes(f'$P{i}B', "utf8") for i in self.channel_numbers]
        keys_encoding_range = [bytes(f'$P{i}R', "utf8") for i in self.channel_numbers]
        add_keys_to_convert_to_int = [b'$NEXTDATA', b'$PAR', b'$TOT']

        keys_to_convert_to_int = keys_encoding_bits + add_keys_to_convert_to_int

        for key in keys_to_convert_to_int:
            value = text[key]
            text[key] = int(value)

        self.annotation.update(text)

        # Update data start segments if needed

        if self._data_start == 0:
            self._data_start = int(text['$BEGINDATA'])
        if self._data_end == 0:
            self._data_end = int(text['$ENDDATA'])


    def read_data(self, file_handle):
        """ Reads the DATA segment of the FCS file. """
        #self._check_assumptions()
        text = self.annotation

        num_events = text[b'$TOT'] # Number of events recorded
        num_pars   = text[b'$PAR'] # Number of parameters recorded

        if text[b'$BYTEORD'].strip() == b'1,2,3,4' or text[b'$BYTEORD'].strip() == b'1,2':
            endian = b'<'
        elif text[b'$BYTEORD'].strip() == b'4,3,2,1' or text[b'$BYTEORD'].strip() == b'2,1':
            endian = b'>'

        #conversion_dict = {'F' : 'f4', 'D' : 'f8', 'I' : 'u'} # matching FCS naming convention with numpy naming convention f4 - 4 byte (32 bit) single precision float
        conversion_dict = {b'F' : b'f', b'D' : b'f', b'I' : b'u'} # matching FCS naming convention with numpy naming convention f4 - 4 byte (32 bit) single precision float

        if text[b'$DATATYPE'] not in conversion_dict.keys():
            raise_parser_feature_not_implemented(bytes(f'$DATATYPE = {text["$DATATYPE"]} is not yet supported.', "utf8"))

        # Calculations to figure out data types of each of parameters
        bytes_per_par_list   = [text[bytes(f'$P{i}B', "utf8")] // 8  for i in self.channel_numbers] # $PnB specifies the number of bits reserved for a measurement of parameter n

        print(bytes_per_par_list)
        # par_numeric_type_list   = [bytes(f'{endian}{conversion_dict[text[b"$DATATYPE"]]}{bytes_per_par}', "utf8") for bytes_per_par in bytes_per_par_list]
        par_numeric_type_list   = [b'%s%s %d' %(endian, conversion_dict[text[b"$DATATYPE"]], bytes_per_par)  for bytes_per_par in bytes_per_par_list]


        print(par_numeric_type_list)
        bytes_per_event = sum(bytes_per_par_list)
        total_bytes = bytes_per_event * num_events

        # Parser for list mode. Here, the order is a list of tuples. where each tuples stores event related information
        file_handle.seek(self._data_start, 0) # Go to the part of the file where data starts

        ##
        # Read in the data
        if len(set(par_numeric_type_list)) > 1:
            # values saved in mixed data formats
            dtype = b','.join(par_numeric_type_list)
            data = numpy.fromfile(file_handle, dtype=dtype, count=num_events)
            data.dtype.names = self.get_channel_names()
        else:
            # values saved in a single data format
            dtype = par_numeric_type_list[0]
            print(dtype)
            data = numpy.fromfile(file_handle, dtype=dtype, count=num_events * num_pars)
            data = data.reshape((num_events, num_pars))
        ##
        # Convert to native byte order 
        # This is needed for working with pandas datastructures
        native_code = '<' if (sys.byteorder == 'little') else '>'
        if endian != native_code:
            # swaps the actual bytes and also the endianness
            print(type(data))
            data = data.byteswap().newbyteorder()

        self._data = data



if __name__ == '__main__':
    mugabe = Reader()
    with open("data.fcs", "rb") as file:
        mugabe.read_header(file)
        mugabe.read_text(file)
        mugabe.read_data(file)
        print(mugabe._data[0][:10])