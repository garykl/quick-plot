"""
Make a class with convenience function for easier plotting.
This class can be called via with statement and does not have to be shown
explicitely.
It supports all methods that are supported by matplotlibs axis object.
Additionally all methods can be given an optional parameter 'filename', which
is then used to save the plot as a file.
"""

import sys


class Plot(object):
    """
    this class mimics matplotlib's axis class. All its methods are
    available.
    Additionally, the with construction can be used.
    """

    def __init__(self, filename=None, figsize=(3, 3), grid=False, **kwargs):
        """ initialize an axes object """
        import matplotlib.pyplot as plt

        self.filename = filename

        self.fig = plt.figure(figsize=figsize)
        self.axe = plt.subplot(111, **kwargs)

        self.axe.grid(grid)

    def __enter__(self):
        """ assign the return value to the with target """
        return self

    def __exit__(self, type, value, traceback):
        """
        since this class is not made for exception handling, but for
        plotting, we do not use the arguments to the method.
        """
        import matplotlib as mlib
        from matplotlib import rc
        import matplotlib.pyplot as plt
        rc('text', usetex=True)
        mlib.rcParams['text.latex.preamble'].append(r'\usepackage{amsmath}')
        mlib.rcParams['text.latex.preamble'].append(r'\usepackage{amssymb}')

        if self.filename is not None:
            self.fig.savefig(self.filename, bbox_inches='tight')
        else:
            plt.show()#block=True)

        plt.close(self.fig)

    def __getattr__(self, name):
        """ this object should act like a matplotlib axis object """
        res = None
        try:
            res = getattr(self.axe, name)
        except:
            import matplotlib.pyplot as plt
            res = getattr(plt, name)
        return res

    def legend(self, legend, plots=None, **kwargs):
        if 'fancybox' not in kwargs:
            kwargs['fancybox'] = True
        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = 7
        return self.axe.legend(legend, **kwargs)

    def plot(self, *args, **kwargs):
        if 'lw' not in kwargs and 'linewidth' not in kwargs:
            kwargs['lw'] = 2
        return self.axe.plot(*args, **kwargs)

    def trajectory(self, x, y, colors, *args, **kwargs):
        self.field(x, y, list(range(len(x))), colors, *args, **kwargs)

    def field(self, x, y, z, colors, *args, **kwargs):
        import numpy as np
        mn = np.amin(z)
        mx = np.amax(z)
        for (x1, x2, y1, y2, zz) in zip(x, x[1:], y, y[1:], z):
            index = int((len(z) - 1) * (zz - mn) / (mx - mn))
            self.axe.plot([x1, x2], [y1, y2], *args,
                    color=colors[index], **kwargs)

    def points(self, *args, **kwargs):
        """ make a plot without a line, but with dots.  """
        if 'marker' not in kwargs:
            kwargs['marker'] = '.'
        return self.axe.plot(*args, ls='None', **kwargs)

    def imshow(self, *args, norm=None, **kwargs):
        """ call matplotlibs imshow with eased usage of norms """
        import matplotlib as mpl
        if norm is not None:
            norm = mpl.colors.Normalize(vmin=norm[0], vmax=norm[1])
        self.axe.imshow(*args, norm=norm, **kwargs)

    def polar_imshow(self, field, *args, **kwargs):

        import numpy as np
        import matplotlib.cm as cm
        def half_periodic_half_space():
            """
            field is a two dimensional grid.
            the first dimension represents the amplitude (ususally),
            the second dimension represents the phase.
            amp_range is the range of the first dimensions.
            return data that can be used as aruments for matplotlibs contour plot.
            """

            (amp_grid, phi_grid) = field.shape
            phis = np.linspace(0, 2 * np.pi * 1, phi_grid + 1)
            amps = np.linspace(0, 3, 3 * amp_grid)
            field_new = np.concatenate(
                    [field, np.transpose([field[:, -1]])], axis=1)
            (a, b) = np.meshgrid(phis, amps)
            empty = np.zeros_like(field_new)
            empty[:, :] = np.nan

            return (a, b, np.concatenate(
                        [empty,
                         field_new,
                         empty],
                        axis=0))

        field = np.array(field)
        (a, b, f) = half_periodic_half_space()
        self.axe.contourf(a, b, f, cmap=cm.nipy_spectral, *args, **kwargs)
        self.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100) * 1, color='k')
        self.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100) * 1.5, color='k')
        self.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100) * 2, color='k')
        self.remove_border()


    def interplot(self, *args, **kwargs):
        """
        interpolate datapoints and plot with doubled discretization.
        regular grid in x is assumed.
        """
        from scipy.interpolate import interp1d
        from numpy import linspace
        x = args[0]
        y = args[1]
        f = interp1d(x, y, kind='cubic')
        x_new = linspace(x[0], x[-1], 2 * len(x))
        self.plot(x_new, f(x_new), *args[2:], **kwargs)

    def colorbar(self, vals, cmap='jet', minimum=None, maximum=None):
        import matplotlib as mpl
        import numpy as np
        if minimum is None:
            minimum = np.amin(vals)
        if maximum is None:
            maximum = np.amax(vals)

        cm = getattr(mpl.cm, cmap)
        norm  = mpl.colors.Normalize(vmin=minimum, vmax=maximum)
        legend = self.fig.add_axes([0.85, 0.25, 0.10, 0.5])
        cb = mpl.colorbar.ColorbarBase(
                legend, cmap=cmap, norm=norm, orientation='vertical')
        return cb

    def set_phaseticks(self):
        import numpy as np
        phases = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        phaselabels = ['$0$', r'$\frac{\pi}{2}$', r'$\pi$',
                       r'$\frac{3\pi}{2}$', r'$2\pi$']
        self.axe.set_xticks(phases)
        self.axe.set_xticklabels(phaselabels)
        self.axe.set_xlim((0, np.pi * 2))

    def remove_xticks(self):
        self.axe.set_xticks([])
        self.axe.set_xticklabels([])

    def remove_yticks(self):
        self.axe.set_yticks([])
        self.axe.set_yticklabels([])

    def remove_border(self):
        self.axe.set_frame_on(False)
        self.remove_xticks()
        self.remove_yticks()

    @staticmethod
    def set_ticks(ticks, tick_function, tick_label_function):
        tick_function(ticks)
        tick_label_function([r'${0}$'.format(t) for t in ticks])

    def set_xticks(self, ticks):
        self.set_ticks(ticks, self.axe.set_xticks, self.axe.set_xticklabels)

    def set_yticks(self, ticks):
        self.set_ticks(ticks, self.axe.set_yticks, self.axe.set_yticklabels)




class WithEncapsulate(object):
    """
    All locals of this class with be transferred to the module's namespace,
    because then, the __getattr__ method can be used for creating methods at
    runtime, which later leads the creation of module functions at runtime.
    """

    def __init__(self, classconstructor, argname):
        """
        save a classconstructor, whose class will be instantiated constantly.
        classconstructor is instantiated with exactly one argument.
        for each function call, that one argument may be provided by an
        optional argument, its default value is None.
        """
        self.classconstructor = classconstructor
        self.argname = argname

    def get_colors(self, num, cmap='jet'):
        import matplotlib.cm as ccmm
        from numpy import linspace
        cm = getattr(ccmm, cmap)
        return cm(linspace(0, 1, num))

    def get_colors_function(self, num, color_func):
        from numpy import linspace, vectorize
        return [color_func(n) for n in linspace(0, 1, num)]

    def color_function(self, interval, cmap='jet'):
        grid = 100
        colors = self.get_colors(grid, cmap=cmap)
        def func(v):
            return colors[
                int((v - interval[0]) / (interval[-1] - interval[0]) * (grid - 1))]
        return func

    def colored(self, data, cmap='jet'):
        return zip(data, self.get_colors(len(data), cmap=cmap))

    def get_greens(self, num, r=0.2, b=0.2):
        from numpy import linspace
        return [(r, g, b) for g in linspace(0.1, 1, num)]

    def get_reds(self, num, g=0.2, b=0.2):
        from numpy import linspace
        return [(r, g, b) for r in linspace(0.1, 1, num)]

    def get_blues(self, num, r=0.2, g=0.2):
        from numpy import linspace
        return [(r, g, b) for b in linspace(0.1, 1, num)]

    def get_greys(self, num):
        from numpy import linspace
        return [(g, g, g) for g in linspace(0.1, 1, num)]

    def golden_ratio(self):
        return 1.618

    def portrait(self, width=3):
        return (width, self.golden_ratio() * width)

    def landscape(self, width=5):
        return (width, width / self.golden_ratio())

    def quadratic(self, size=5):
        return (size, size)

    def quickplot(self, d):
        with Plot() as p:
            p.plot(d)

    def tex(self, string):
        return r'${0}$'.format(string)

    def __getattr__(self, name):
        """ make methods of QuickPlot available as method of this wrapper """
        if name == self.classconstructor.__name__:
            return self.classconstructor
        def hfunc(*args, **kwargs):
            value = None
            if self.argname in kwargs:
                value = kwargs[self.argname]
                del kwargs[self.argname]
            with self.classconstructor(value) as plt:
                getattr(plt, name)(*args, **kwargs)
        return hfunc

### release QuickPlot and all the matplotlib axis methods as functions.
sys.modules[__name__] = WithEncapsulate(Plot, 'filename')
