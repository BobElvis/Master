import timer
t = timer.SimpleTimer()
import numpy as np


class BackgroundDrawer:
    def __init__(self, artist):
        self.ax = artist
        self.gc = None
        self.res = None
        self.res_prev = None
        self.i = 0

    def reset(self):
        # resize, new limits
        self.res = None
        self.i = 0

    def draw(self, renderer):
        ax = self.ax
        # if not visible, declare victory and return
        if not ax.get_visible():
            ax.stale = False
            return

        # for empty images, there is nothing to draw!
        if ax.get_array().size == 0:
            ax.stale = False
            return

        gc = renderer.new_gc()
        ax._set_gc_clip(gc)
        gc.set_alpha(ax.get_alpha())
        gc.set_url(ax.get_url())
        gc.set_gid(ax.get_gid())

        if (ax._check_unsampled_image(renderer) and
                ax.get_transform().is_affine):
            ax._draw_unsampled_image(renderer, gc)
        else:
            if self.res is None:
                res = ax.make_image(renderer, renderer.get_image_magnification())
                if self.i > 0:
                    self.res = res
            else:
                res = self.res
            im, l, b, trans = res
            if im is not None:
                renderer.draw_image(gc, l, b, im)
        gc.restore()
        ax.stale = False
        self.i = self.i + 1