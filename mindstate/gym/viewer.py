import itertools

import pyglet


class SimpleAttentiveImageViewer(object):
    def __init__(self, attention_size, display=None):
        self.attention_size = attention_size
        self.window = None
        self.isopen = False
        self.display = display
        self.rect = (0, 0, 100, 100)

    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(
                width=2 * width + self.attention_size, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
        a_x, a_y, a_w, a_h = self.rect_to_pyglet(self.rect)
        attention_image = image.get_region(*self.rect_to_pyglet(self.rect))
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0,0)
        # show in corner
        o_w, o_h = attention_image.texture.width, attention_image.texture.height
        attention_image.texture.width = self.attention_size
        attention_image.texture.height = self.attention_size
        attention_image.blit(self.width * 2, self.height - self.attention_size)
        # add to overall map
        attention_image.texture.width = o_w
        attention_image.texture.height = o_h
        attention_image.blit(self.width + a_x, a_y)

        pyglet.gl.glColor4f(1., 1., 1., 1.)
        pyglet.graphics.draw(
            4, pyglet.gl.GL_LINE_LOOP,
            ('v2i', list(itertools.chain(*self.rect_to_coords(self.rect))))
        )
        self.window.flip()

    def rect_to_coords(self, rect):
        #print(rect)
        (x, y, w, h) = self.rect_to_pyglet(rect)
        return [
            (x, y), (x + w, y), (x + w, y + h), (x, y + h)
        ]

    def rect_to_pyglet(self, rect):
        (x, y, w, h) = rect
        return (x, self.height - y - h, w, h)

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()
