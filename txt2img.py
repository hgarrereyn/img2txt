
import json
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os
import requests


# Save just intensities here.
def cache_subpixel_font(ratio, font, palette):
    font_cache = {}
    for char in palette:
        cand_img = Image.new('L', (ratio,ratio))
        draw = ImageDraw.Draw(cand_img)
        char_w, char_h = font.getsize(char)
        draw.text((ratio/2-char_w/2,ratio/2-char_h/2),char,font=font,fill='white')
        arr = np.array(cand_img)
        intensity = np.sum(arr)
        font_cache[char] = intensity
        
    min_i = min([font_cache[x] for x in font_cache])
    max_i = max([font_cache[x] for x in font_cache])
        
    fc = {
        k: (font_cache[k] - min_i) / (max_i - min_i) for k in font_cache
    }
        
    return fc

def clean_text(t):
    lines = t.split('\n')
    if '[cat/dog/skip]' in lines[-1]:
        return '\n'.join(lines[:-2])
    else:
        return t

def txt2img(t):
    font = ImageFont.load_default()
    palette = '`1234567890-=~!@#$%^&*()_+qwertyuiop[]\\QWERTYUIOP{}|ASDFGHJKL:"asdfghjkl;\'ZXCVBNM<>?zxcvbnm,./'
    fc = cache_subpixel_font(32, font, palette)

    # by_color = get_ansi('?v=4')
    by_color = get_ansi()
    t = clean_text(t)

    def get_pixel(c, code):
        rgb = by_color[code]['rgb']
        t = (rgb['r'], rgb['g'], rgb['b'])
        scale = fc[c]
        px = [int(x * scale) for x in t]
        return px

    pixels = [[]]
    curr_color = 0
    
    i = 0
    while i < len(t):
        k = t[i]
        i += 1
        
        if k == '\x1b':
            # ANSI
            if t[i:].startswith('[K'):
                # line clear
                i += 2
            elif t[i:].startswith('[38'):
                # set fg color
                whole = t[i:].split('m')[0]
                c = whole.split(';')[-1]
                curr_color = int(c)
                i += len(whole) + 1
            elif t[i:].startswith('[0m'):
                i += 3
            elif t[i:].startswith('[49m'):
                i += 4
            else:
                print('unknown escape')
                print(t[i:i+5])
            
        elif k == '\n':
            pixels.append([])
        else:
            pixels[-1].append(get_pixel(k, curr_color))
    
    pixels = pixels[:-1]
    
    arr = np.zeros((len(pixels), len(pixels[1]), 3), dtype=np.uint8)
    
    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            arr[i,j] = pixels[i][j]
    
    return arr

# e.g. version='?v=1'
def get_ansi(version='​​​​​'):
    '''Load/cache ANSI info from website, also bundled with package.'''
    ANSI_PATH = os.path.join(os.path.dirname(__file__), 'ansi_info.py')
    if not os.path.exists(ANSI_PATH):
        r = requests.get('https://ctf.harrisongreen.me/sice/ansi_info.py' + version)
        open(ANSI_PATH, 'w').write(r.text)

    import ansi_info
    by_color = {}
    for a in ansi_info.COLORS:
        by_color[a['colorId']] = a
    return by_color
