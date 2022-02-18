# Author: aqeelanwar 
# Created: 6 March,2020, 11:29 AM
# Email: aqeel.anwar@gatech.edu

import imageio
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from PIL import Image
from fire import Fire
from matplotlib.colors import ListedColormap
from tqdm import tqdm


def fig2img(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buffer.shape = (w, h, 4)
    buffer = np.roll(buffer, 3, axis=2)
    w, h, d = buffer.shape
    img = Image.frombytes("RGBA", (w, h), buffer.tobytes())
    return img


def run_animation(output_sz, input_sz, duration, layer_type, padding_sz, actual_padding_sz, stride, actual_stride,
                  kernel_sz, kernel, gif_name, padded_input, fig, axes):
    annot = False
    if output_sz == int(output_sz):
        output_sz = int(output_sz)
        y = np.ones(shape=(output_sz, output_sz))
        # sns.heatmap(padded_input, yticklabels=False, xticklabels=False, ax=axes[0], annot=True, cbar = False, linewidth=0.5, cmap=ListedColormap(['#A6A6A6', '#DD8047', '#94B6D2']))
        # Filter sliding input
        img = []

        for i in tqdm(range(output_sz)):
            for j in range(output_sz):
                axes[0].clear()
                axes[1].clear()
                plt.suptitle(f"Type: {layer_type}  -  Stride: {actual_stride}  Padding: {actual_padding_sz}",
                             fontsize=20, fontname='cmr10')
                # Input and Kernel
                array_cmap = np.asarray(['#DD8047', '#CD8B67', '#A6A6A6', '#94B6D2'])
                cmap_indices = np.asarray(np.sort(list(set(np.unique(kernel + padded_input)))) + 2)
                cmap_val = array_cmap[cmap_indices]
                cmap = ListedColormap(cmap_val)

                plot_vals = kernel + padded_input
                if set(np.unique(kernel + padded_input)) == set([-2, 0, 1]):
                    plot_vals = np.sign(plot_vals)

                sns.heatmap(plot_vals, yticklabels=False, xticklabels=False, ax=axes[0],
                            annot=annot, cbar=False, linewidth=0.5, cmap=cmap)

                axes[0].set_xlabel('Input', fontdict={'fontsize': 16, 'fontname': 'cmr10'})
                # Output
                y[i, j] = 0
                sns.heatmap(np.sign(y), yticklabels=False, xticklabels=False, ax=axes[1], annot=annot, cbar=False,
                            linewidth=0.5, cmap=ListedColormap(['#A5Ab81', '#DBDDCD']))

                axes[1].set_xlabel('Output', fontdict={'fontsize': 16, 'fontname': 'cmr10'})
                img.append(fig2img(fig))
                plt.close(fig)
                shift_num = stride
                if (j * (stride) + kernel_sz) >= (input_sz + 2 * padding_sz):
                    shift_num = kernel_sz + (stride - 1) * (input_sz + 2 * padding_sz)
                kernel = np.roll(kernel, shift_num)

        imageio.mimsave(gif_name, img, duration=duration)
    else:
        print('Set the parameters')


def main(input_sz=3,
         kernel_sz=3,
         stride=2,
         padding_sz=1,
         duration=1,
         layer_type='transposed_conv'
         ):
    matplotlib.use('Agg')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # fig.tight_layout()

    # Set the parameters
    # layer_type = 'conv'

    gif_name = layer_type + '_S' + str(stride) + 'P' + str(padding_sz) + '.gif'
    actual_stride = stride
    actual_padding_sz = padding_sz

    x = np.ones(shape=(input_sz, input_sz), dtype=int)

    if layer_type == 'conv':
        # For conv:
        output_sz = (input_sz + 2 * padding_sz - kernel_sz) / stride + 1
        new_input = x

    elif layer_type == 'transposed_conv':
        # For transposed_conv
        stride = 1
        output_sz = (input_sz - 1) * actual_stride + kernel_sz - 2 * padding_sz

        padding_sz = kernel_sz - padding_sz - 1
        zero_insertion = actual_stride - 1

        pos = np.arange(1, input_sz)
        pos = np.repeat(pos, zero_insertion)
        zero_inserted_input = np.insert(x, pos, 0, axis=1)
        zero_inserted_input = np.insert(zero_inserted_input, pos, 0, axis=0)

        new_input = zero_inserted_input

    p1 = np.zeros(shape=(padding_sz, 2 * padding_sz + new_input.shape[0]), dtype=int)
    p21 = np.zeros(shape=(new_input.shape[0], padding_sz), dtype=int)
    padded_input = np.block([[p1], [p21, new_input, p21], [p1]])

    k11 = np.ones(shape=(kernel_sz, kernel_sz), dtype=int)
    k12 = np.zeros(shape=(kernel_sz, new_input.shape[0] + 2 * padding_sz - kernel_sz), dtype=int)
    k2 = np.zeros(shape=(new_input.shape[0] + 2 * padding_sz - kernel_sz, 2 * padding_sz + new_input.shape[0]),
                  dtype=int)

    kernel = -2 * np.block([[k11, k12], [k2]])

    run_animation(output_sz, new_input.shape[0], duration, layer_type, padding_sz, actual_padding_sz, stride,
                  actual_stride, kernel_sz, kernel, gif_name, padded_input, fig, axes)


if __name__ == "__main__":
    Fire(main)
