# %%
from cyclegan import CycleGAN
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic('load_ext', 'autoreload')  # noqa
    get_ipython().run_line_magic('autoreload', '2')  # noqa
    print('Loaded extensions')
except Exception:
    pass

# %%
gan = CycleGAN(dataset_name='base2basent', input_shape=(128, 128, 3))

# %%
gan.load_model()

# %%
imgs_A = gan.data_loader.load_data(domain="A", batch_size=1, is_testing=True)

# %%
plt.imshow(imgs_A[0])

# %%
# Translate images to the other domain
fake_B = gan.g_AB.predict(imgs_A)

# %%
plt.imshow(fake_B[0])
