from __future__ import print_function

import sys
import wave

import pyaudio
from io import StringIO

import alsaaudio
import colorama
import numpy as np

import math

from reedsolo import RSCodec, ReedSolomonError
from termcolor import cprint
from pyfiglet import figlet_format

HANDSHAKE_START_HZ = 4096
HANDSHAKE_END_HZ = 5120 + 1024

START_HZ = 1024
STEP_HZ = 256
BITS = 4

FEC_BYTES = 4



#frames = []

def stereo_to_mono(input_file, output_file):
    inp = wave.open(input_file, 'r')
    params = list(inp.getparams())
    params[0] = 1 # nchannels
    params[3] = 0 # nframes

    out = wave.open(output_file, 'w')
    out.setparams(tuple(params))

    frame_rate = inp.getframerate()
    frames = inp.readframes(inp.getnframes())
    data = np.fromstring(frames, dtype=np.int16)
    left = data[0::2]
    out.writeframes(left.tostring())

    inp.close()
    out.close()

def yield_chunks(input_file, interval):
    wav = wave.open(input_file)
    frame_rate = wav.getframerate()

    chunk_size = int(round(frame_rate * interval))
    total_size = wav.getnframes()

    while True:
        chunk = wav.readframes(chunk_size)
        if len(chunk) == 0:
            return

        yield frame_rate, np.fromstring(chunk, dtype=np.int16)

def dominant(frame_rate, chunk):
    w = np.fft.fft(chunk)
    freqs = np.fft.fftfreq(len(chunk))
    peak_coeff = np.argmax(np.abs(w))
    peak_freq = freqs[peak_coeff]
    return abs(peak_freq * frame_rate) # in Hz

def match(freq1, freq2):
    return abs(freq1 - freq2) < 20

def match1(freq1, freq2):
    return abs(freq1 - freq2) == 520

def decode_bitchunks(chunk_bits, chunks):
    out_bytes = []

    next_read_chunk = 0
    next_read_bit = 0

    byte = 0
    bits_left = 8
    while next_read_chunk < len(chunks):
        can_fill = chunk_bits - next_read_bit
        to_fill = min(bits_left, can_fill)
        offset = chunk_bits - next_read_bit - to_fill
        byte <<= to_fill
        shifted = chunks[next_read_chunk] & (((1 << to_fill) - 1) << offset)
        byte |= shifted >> offset;
        bits_left -= to_fill
        next_read_bit += to_fill
        if bits_left <= 0:

            out_bytes.append(byte)
            byte = 0
            bits_left = 8

        if next_read_bit >= chunk_bits:
            next_read_chunk += 1
            next_read_bit -= chunk_bits

    return out_bytes

def decode_file(input_file, speed):
    wav = wave.open(input_file)
    if wav.getnchannels() == 2:
        mono = StringIO()
        stereo_to_mono(input_file, mono)

        mono.seek(0)
        input_file = mono
    wav.close()

    offset = 0
    for frame_rate, chunk in yield_chunks(input_file, speed / 2):
        dom = dominant(frame_rate, chunk)
        print("{} => {}".format(offset, dom))
        offset += 1

def extract_packet(freqs):
    freqs = freqs[::2]
    bit_chunks = [int(round((f - START_HZ) / STEP_HZ)) for f in freqs]
    bit_chunks = [c for c in bit_chunks[1:] if 0 <= c < (2 ** BITS)]
    return bytearray(decode_bitchunks(BITS, bit_chunks))

def display(s):
    cprint(figlet_format(s.replace(' ', '   '), font='doom'), 'yellow')

def listen_linux(frame_rate=44100, interval=0.1):

    mic = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device="default")
    mic.setchannels(1)
    mic.setrate(44100)
    mic.setformat(alsaaudio.PCM_FORMAT_S16_LE)

    num_frames = int(round((interval / 2) * frame_rate))
    mic.setperiodsize(num_frames)
    print("start...")

    in_packet = False
    packet = []
    count = 0
    ccount = 0    
 
    while True:
        l, data = mic.read()
        if not l:
            continue

        chunk = np.fromstring(data, dtype=np.int16)
        dom = dominant(frame_rate, chunk)
    #    print(packet)
    #    print("dom : ", dom)

        if in_packet and match(dom, HANDSHAKE_END_HZ):
            byte_stream = extract_packet(packet)
            try:
                byte_stream = RSCodec(FEC_BYTES).decode(byte_stream)
                byte_stream = byte_stream.decode("utf-8")
                my_school_number = byte_stream[0] + byte_stream[1]+byte_stream[2]+byte_stream[3]+byte_stream[4]+byte_stream[5]+byte_stream[6]+byte_stream[7]+byte_stream[8]
                hello = byte_stream[10]+byte_stream[11]+byte_stream[12]+byte_stream[13]+byte_stream[14]
#                print(hello)
#                print(my_school_number)

                if my_school_number == "201701974":
                       print(byte_stream)
                       display(hello)#byte_stream)
                       #byte_stream1 = hello.encode()
                       #p = pyaudio.PyAudio()
                       duration = 1.5
                       frame = []
                       for value in hello:
                           value = int(ord(value))
                           max_1 = bin((value & 0b10000000)>>7)
                           max_2 = bin((value & 0b01000000)>>6)
                           max_3 = bin((value & 0b00100000)>>5)
                           max_4 = bin((value & 0b00010000)>>4)
                           max_5 = bin((value & 0b00001000)>>3)
                           max_6 = bin((value & 0b00000100)>>2)
                           max_7 = bin((value & 0b00000010)>>1)
                           max_8 = bin((value & 0b00000001))
#                           print(maxx + " " + minn)
                           frame.append(int(max_1,2)*STEP_HZ + START_HZ)
                           frame.append(int(max_2,2)*STEP_HZ + START_HZ)
                           frame.append(int(max_3,2)*STEP_HZ + START_HZ)
                           frame.append(int(max_4,2)*STEP_HZ + START_HZ)
                           frame.append(int(max_5,2)*STEP_HZ + START_HZ)
                           frame.append(int(max_6,2)*STEP_HZ + START_HZ)
                           frame.append(int(max_7,2)*STEP_HZ + START_HZ)
                           frame.append(int(max_8,2)*STEP_HZ + START_HZ)
                       for value in frame:
                           p = pyaudio.PyAudio()
                           stream = p.open(format=pyaudio.paFloat32, channels=1, rate = 44100, output = True)
                           samples = (np.sin(2*np.pi*value*np.arange(0,duration,1/44100)))
                           stream.write(samples)

            except ReedSolomonError as e:
                pass
                #print("{}: {}".format(e, byte_stream))

            packet = []
            in_packet = False
            
        elif in_packet:
            packet.append(dom)
            
            
        elif match(dom, HANDSHAKE_START_HZ):
            print("start Handshake!")
            in_packet = True
           

if __name__ == '__main__':
    colorama.init(strip=not sys.stdout.isatty())

    #decode_file(sys.argv[1], float(sys.argv[2]))
    listen_linux()

