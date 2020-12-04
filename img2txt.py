#!/usr/bin/env python
# -*- coding: utf-8 -*-

# DISPLAY=localhost:0.0 python3 img2txt/img2txt.py --antialias --dither --ansi --targetAspect 0.5 --color --maxLen=64 cat.jpg --subpixel 32 --subpixel-font consola.ttf --gpu

("""
Usage:
  img2txt.py <imgfile> [--maxLen=<n>] [--fontSize=<n>] [--color] [--ansi]"""
    """[--bgcolor=<#RRGGBB>] [--targetAspect=<n>] [--antialias] [--dither] [--subpixel=<n>] [--subpixel-font=<font.ttf>] [--gpu]
  img2txt.py (-h | --help)

Options:
  -h --help             show this screen.
  --ansi                output an ANSI rendering of the image
  --color               output a colored HTML rendering of the image.
  --antialias           causes any resizing of the image to use antialiasing
  --dither              dither the colors to web palette. Useful when converting 
                        images to ANSI (which has a limited color palette)  
  --fontSize=<n>        sets font size (in pixels) when outputting HTML,
                        default: 7
  --maxLen=<n>          resize image so that larger of width or height matches
                        maxLen, default: 100px
  --bgcolor=<#RRGGBB>   if specified, is blended with transparent pixels to
                        produce the output. In ansi case, if no bgcolor set, a
                        fully transparent pixel is not drawn at all, partially
                        transparent pixels drawn as if opaque
  --targetAspect=<n>    resize image to this ratio of width to height. Default is 
                        1.0 (no resize). For a typical terminal where height of a 
                        character is 2x its width, you might want to try 0.5 here
  --subpixel=<n>        perform subpixel rendering, with scale factor n
  --subpixel-font=<f>   font to assume for subpixel rendering. must be TTF format
  --gpu                 perform subpixel rendering with CUDA on GPU.
""")

import sys
from docopt import docopt
from PIL import Image, ImageFont, ImageDraw
import numpy
import ansi
from graphics_util import alpha_blend

def HTMLColorToRGB(colorstring):
    """ convert #RRGGBB to an (R, G, B) tuple """
    colorstring = colorstring.strip()
    if colorstring[0] == '#':
        colorstring = colorstring[1:]
    if len(colorstring) != 6:
        raise ValueError(
            "input #{0} is not in #RRGGBB format".format(colorstring))
    r, g, b = colorstring[:2], colorstring[2:4], colorstring[4:]
    r, g, b = [int(n, 16) for n in (r, g, b)]
    return (r, g, b)


def generate_HTML_for_image(pixels, width, height):

    string = ""
    # first go through the height,  otherwise will rotate
    for h in range(height):
        for w in range(width):

            rgba = pixels[w, h]

            # TODO - could optimize output size by keeping span open until we
            # hit end of line or different color/alpha
            #   Could also output just rgb (not rgba) when fully opaque - if
            #   fully opaque is prevalent in an image
            #   those saved characters would add up
            string += ("<span style=\"color:rgba({0}, {1}, {2}, {3});\">"
                       "▇</span>").format(
                rgba[0], rgba[1], rgba[2], rgba[3] / 255.0)

        string += "\n"

    return string


def generate_grayscale_for_image(pixels, width, height, bgcolor):

    # grayscale
    color = "MNHQ$OC?7>!:-;. "

    string = ""
    # first go through the height,  otherwise will rotate
    for h in range(height):
        for w in range(width):

            rgba = pixels[w, h]

            # If partial transparency and we have a bgcolor, combine with bg
            # color
            if rgba[3] != 255 and bgcolor is not None:
                rgba = alpha_blend(rgba, bgcolor)

            # Throw away any alpha (either because bgcolor was partially
            # transparent or had no bg color)
            # Could make a case to choose character to draw based on alpha but
            # not going to do that now...
            rgb = rgba[:3]

            string += color[int(sum(rgb) / 3.0 / 256.0 * 16)]

        string += "\n"

    return string


def load_and_resize_image(img, antialias, maxLen, aspectRatio, align=None):

    if aspectRatio is None:
        aspectRatio = 1.0

    # force image to RGBA - deals with palettized images (e.g. gif) etc.
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # need to change the size of the image?
    if maxLen is not None or aspectRatio != 1.0:

        native_width, native_height = img.size

        new_width = native_width
        new_height = native_height

        # First apply aspect ratio change (if any) - just need to adjust one axis
        # so we'll do the height.
        if aspectRatio != 1.0:
            new_height = int(float(aspectRatio) * new_height)

        # Now isotropically resize up or down (preserving aspect ratio) such that 
        # longer side of image is maxLen 
        if maxLen is not None:
            rate = float(maxLen) / max(new_width, new_height)
            new_width = int(rate * new_width)  
            new_height = int(rate * new_height)

        if align is not None:
            new_width -= new_width % align
            new_height -= new_height % align


        if native_width != new_width or native_height != new_height:
            img = img.resize((new_width, new_height), Image.ANTIALIAS if antialias else Image.NEAREST)

    return img


def floydsteinberg_dither_to_web_palette(img):

    # Note that alpha channel is thrown away - if you want to keep it you need to deal with it yourself
    #
    # Here's how it works:
    #   1. Convert to RGB if needed - we can't go directly from RGBA because Image.convert will not dither in this case
    #   2. Convert to P(alette) mode - this lets us kick in dithering.
    #   3. Convert back to RGBA, which is where we want to be 
    #
    # Admittedly converting back and forth requires more memory than just dithering directly
    # in RGBA but that's how the library works and it isn't worth writing it ourselves
    # or looking for an alternative given current perf needs.

    if img.mode != 'RGB': 
        img = img.convert('RGB')     
    img = img.convert(mode="P", matrix=None, dither=Image.FLOYDSTEINBERG, palette=Image.WEB, colors=256)
    img = img.convert('RGBA')    
    return img


def dither_image_to_web_palette(img, bgcolor):
    
    if bgcolor is not None:
        # We know the background color so flatten the image and bg color together, thus getting rid of alpha
        # This is important because as discussed below, dithering alpha doesn't work correctly.
        img = Image.alpha_composite(Image.new("RGBA", img.size, bgcolor), img)  # alpha blend onto image filled with bgcolor
        dithered_img = floydsteinberg_dither_to_web_palette(img)    
    else:
     
        """
        It is not possible to correctly dither in the presence of transparency without knowing the background
        that the image will be composed onto. This is because dithering works by propagating error that is introduced 
        when we select _available_ colors that don't match the _desired_ colors. Without knowing the final color value 
        for a pixel, it is not possible to compute the error that must be propagated FROM it. If a pixel is fully or 
        partially transparent, we must know the background to determine the final color value. We can't even record
        the incoming error for the pixel, and then later when/if we know the background compute the full error and 
        propagate that, because that error needs to propagate into the original color selection decisions for the other
        pixels. Those decisions absorb error and are lossy. You can't later apply more error on top of those color
        decisions and necessarily get the same results as applying that error INTO those decisions in the first place.   
        
        So having established that we could only handle transparency correctly at final draw-time, shouldn't we just 
        dither there instead of here? Well, if we don't know the background color here we don't know it there either. 
        So we can either not dither at all if we don't know the bg color, or make some approximation. We've chosen 
        the latter. We'll handle it here to make the drawing code simpler. So what is our approximation? We basically
        just ignore any color changes dithering makes to pixels that have transparency, and prevent any error from being 
        propagated from those pixels. This is done by setting them all to black before dithering (using an exact-match 
        color in Floyd Steinberg dithering with a web-safe-palette will never cause a pixel to receive enough inbound error
        to change color and thus will not propagate error), and then afterwards we set them back to their original values. 
        This means that transparent pixels are essentially not dithered - they ignore (and absorb) inbound error but they
        keep their original colors. We could alternately play games with the alpha channel to try to propagate the error 
        values for transparent pixels through to when we do final drawing but it only works in certain cases and just isn't 
        worth the effort (which involves writing the dithering code ourselves for one thing).
        """
        
        # Force image to RGBA if it isn't already - simplifies the rest of the code    
        if img.mode != 'RGBA': 
            img = img.convert('RGBA')    

        rgb_img = img.convert('RGB')    
    
        orig_pixels = img.load()
        rgb_pixels = rgb_img.load()
        width, height = img.size

        for h in range(height):    # set transparent pixels to black
            for w in range(width):
                if (orig_pixels[w, h])[3] != 255:    
                    rgb_pixels[w, h] = (0, 0, 0)   # bashing in a new value changes it!

        dithered_img = floydsteinberg_dither_to_web_palette(rgb_img)    

        dithered_pixels = dithered_img.load() # must do it again
        
        for h in range(height):    # restore original RGBA for transparent pixels
            for w in range(width):
                if (orig_pixels[w, h])[3] != 255:    
                    dithered_pixels[w, h] = orig_pixels[w, h]   # bashing in a new value changes it!

    return dithered_img

import colorsys

def char_for_rgb(rgba):
    color = "@#x."
    # If partial transparency and we have a bgcolor, combine with bg
    # color
    if rgba[3] != 255 and bgcolor is not None:
        rgba = alpha_blend(rgba, bgcolor)

    # Throw away any alpha (either because bgcolor was partially
    # transparent or had no bg color)
    # Could make a case to choose character to draw based on alpha but
    # not going to do that now...
    rgb = rgba[:3]

    luminosity =  0.21*rgb[0] + 0.72*rgb[1] + 0.07*rgb[2]
    h,s,v = colorsys.rgb_to_hsv(*rgb)
    import math
    vv = math.log(v+1,2)/math.log(256.0,2) # humans perceive shit logarithmically
    index = int((0.999-min(vv,0.999)) * (len(color)-1))
    return color[index]

def cache_subpixel_font(ratio, font, palette):
    font_cache = {}
    for char in palette:
        cand_img = Image.new('L', (ratio,ratio))
        draw = ImageDraw.Draw(cand_img)
        char_w, char_h = font.getsize(char)
        draw.text((ratio/2-char_w/2,ratio/2-char_h/2),char,font=font,fill='white')
        font_cache[char] = cand_img
    return font_cache

def subpixel_rendering(bigly, ratio, font_cache, use_gpu=True,brightness_gain=32):
    if use_gpu:
        import cupy
    else:
        import numpy as cupy

    assert bigly.width%ratio == 0 and bigly.height%ratio == 0
    img_w, img_h = bigly.width, bigly.height
    out_w, out_h = bigly.width//ratio, bigly.height//ratio
    # we need to cast to int here to avoid overflow issues
    bigly=cupy.asarray(bigly.convert('L')).astype(cupy.int16)
    bigly=bigly+brightness_gain
    # dims: (char, x, y)
    # bigly=cupy.broadcast_to(bigly, (len(font_cache), img_h, img_w))
    # print(bigly.shape)
    chars = list(font_cache)
    patches = cupy.asarray([cupy.asarray(font_cache[c]).astype(cupy.int16) for c in chars]) # upcast to int to avoid overflow issues
    # print (patches.shape)
    b = cupy.tile(patches, (out_h,out_w)) # repeat patches on x,y dimension to tile the scaled up image. 1 tile for each pixel
    # print(b.shape)
    # print(b)
    diffs = abs(bigly - b)
    # print(diffs.shape)
    # split into individual tiles (output pixels)
    # print(bigly)
    # print('-')
    # print(b[31])
    # print('=')
    # print(diffs[31])
    diffs = diffs.reshape(len(chars),out_h,ratio,out_w,ratio).swapaxes(2,3)
    # print(diffs[31,0,0])
    # exit()
    diffs = diffs.reshape(len(chars),out_h,out_w,ratio*ratio) # flatten innermost (each tile)
    diffs = cupy.sum(diffs, axis=3) # manhattan norms per tile
    # print(diffs.shape)
    # print(diffs)
    best_char = diffs.argmin(axis=0)

    if use_gpu:
        cupy.cuda.Stream.null.synchronize()

    # print(best_char)
    chars = [[chars[int(best_char[j,i])] for i in range(out_w)] for j in range(out_h)]
    # print (chars)

    # exit()

    return chars

def img2txt(img, maxLen=100.0, clr=False, do_ansi=False, fontSize=7, bgcolor=None, antialias=False, 
    dither=False, target_aspect_ratio=1.0, use_gpu=False, subpixel_ratio=None, subpixel_font=None):
    '''Perform img2txt conversion.

    Args:
    - img: a PIL image or numpy array
    - *: see docstring
    '''

    output = ''

    if type(img) is numpy.ndarray:
        img = Image.fromarray(img)

    img = load_and_resize_image(img, antialias, maxLen, target_aspect_ratio)

    # Dither _after_ resizing
    if dither:
        img = dither_image_to_web_palette(img, bgcolor)

    # get pixels
    pixel = img.load()

    width, height = img.size

    if do_ansi:

        # Since the "current line" was not established by us, it has been
        # filled with the current background color in the
        # terminal. We have no ability to read the current background color
        # so we want to refill the line with either
        # the specified bg color or if none specified, the default bg color.
        if bgcolor is not None:
            # Note that we are making the assumption that the viewing terminal
            # supports BCE (Background Color Erase) otherwise we're going to
            # get the default bg color regardless. If a terminal doesn't
            # support BCE you can output spaces but you'd need to know how many
            # to output (too many and you get linewrap)
            fill_string = ansi.getANSIbgstring_for_ANSIcolor(
                ansi.getANSIcolor_for_rgb(bgcolor))
        else:
            # reset bg to default (if we want to support terminals that can't
            # handle this will need to instead use 0m which clears fg too and
            # then when using this reset prior_fg_color to None too
            fill_string = "\x1b[49m"
        fill_string += "\x1b[K"          # does not move the cursor
        output += fill_string

        if subpixel_ratio:
            subpixel_ratio = int(subpixel_ratio)
            img = load_and_resize_image(img, antialias, maxLen*subpixel_ratio, target_aspect_ratio,align=subpixel_ratio)

            if subpixel_font is not None:
                font = ImageFont.truetype(subpixel_font, subpixel_ratio)
            else:
                font = ImageFont.load_default()
            palette = '`1234567890-=~!@#$%^&*()_+qwertyuiop[]\\QWERTYUIOP{}|ASDFGHJKL:"asdfghjkl;\'ZXCVBNM<>?zxcvbnm,./'
            font_cache = cache_subpixel_font(subpixel_ratio, font, palette)
            chars = subpixel_rendering(img, subpixel_ratio,font_cache,use_gpu)

            func = lambda pixels, x, y: (chars[y][x], pixel[x, y])
        else:
            func = lambda pixels, x, y: (char_for_rgb(pixel[x,y]), pixel[x, y])

        output += ansi.generate_ANSI_from_pixels(pixel, width, height, bgcolor, get_pixel_func=func)[0]

        # Undo residual color changes, output newline because
        # generate_ANSI_from_pixels does not do so
        # removes all attributes (formatting and colors)
        output += "\x1b[0m\n"
    else:

        if clr:
            # TODO - should handle bgcolor - probably by setting it as BG on
            # the CSS for the pre
            string = generate_HTML_for_image(pixel, width, height)
        else:
            string = generate_grayscale_for_image(
                pixel, width, height, bgcolor)

        # wrap with html

        template = """<!DOCTYPE HTML>
        <html>
        <head>
          <meta http-equiv="content-type" content="text/html; charset=utf-8" />
          <style type="text/css" media="all">
            pre {
              white-space: pre-wrap;       /* css-3 */
              white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
              white-space: -pre-wrap;      /* Opera 4-6 */
              white-space: -o-pre-wrap;    /* Opera 7 */
              word-wrap: break-word;       /* Internet Explorer 5.5+ */
              font-family: 'Menlo', 'Courier New', 'Consola';
              line-height: 1.0;
              font-size: %dpx;
            }
          </style>
        </head>
        <body>
          <pre>%s</pre>
        </body>
        </html>
        """

        html = template % (fontSize, string)
        output += html

    return output

def main(options):
    imgname = options['<imgfile>']
    maxLen = options['--maxLen']
    clr = options['--color']
    do_ansi = options['--ansi']
    fontSize = options['--fontSize']
    bgcolor = options['--bgcolor']
    antialias = options['--antialias']
    dither = options['--dither']
    target_aspect_ratio = options['--targetAspect']
    use_gpu = options['--gpu']
    subpixel_ratio = options['--subpixel']
    subpixel_font = options['--subpixel-font']

    try:
        maxLen = float(maxLen)
    except:
        maxLen = 100.0   # default maxlen: 100px

    try:
        fontSize = int(fontSize)
    except:
        fontSize = 7

    try:
        # add fully opaque alpha value (255)
        bgcolor = HTMLColorToRGB(bgcolor) + (255, )
    except:
        bgcolor = None

    try:
        target_aspect_ratio = float(target_aspect_ratio)
    except:
        target_aspect_ratio = 1.0   # default target_aspect_ratio: 1.0

    try:
        img = Image.open(imgname)
    except IOError:
        exit("File not found: " + imgname)

    output = img2txt(img, maxLen, clr, do_ansi, fontSize, bgcolor, antialias, dither, 
        target_aspect_ratio, use_gpu, subpixel_ratio, subpixel_font)

    return output


if __name__ == '__main__':

    dct = docopt(__doc__)
    output = main(dct)
    sys.stdout.write(output)
    sys.stdout.flush()
