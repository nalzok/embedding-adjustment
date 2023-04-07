from typing import Tuple
from functools import partial

import torch
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, FlaxViTModel, logging
import jax
import jax.random as jr
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.jax_utils import replicate, unreplicate
import optax
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from tqdm import tqdm, trange
import numpy as np


def main(num_epochs: int, batch_size: int, learning_rate: float):
    key = jr.PRNGKey(42)
    generator = torch.Generator().manual_seed(42)
    device_count = jax.local_device_count()

    dataset = get_dataset(dataset="waterbirds", download=True)
    assert dataset is not None

    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
    )
    test_data = dataset.get_subset(
        "test",
        transform=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
    )

    train_loader = get_train_loader(
        "standard", train_data, batch_size, generator=generator
    )
    assert train_loader is not None
    test_loader = get_eval_loader(
        "standard", test_data, batch_size, generator=generator
    )
    assert test_loader is not None

    pretrained = "google/vit-base-patch16-224-in21k"
    embedding_dim = 768

    logging.set_verbosity_error()
    image_processor = AutoImageProcessor.from_pretrained(pretrained)
    vit = FlaxViTModel.from_pretrained(pretrained)
    assert isinstance(vit, FlaxViTModel)
    vit = jax.pmap(vit)

    linear = nn.Dense(4)
    key_init, key = jax.random.split(key)
    specimen = jnp.empty((1, embedding_dim))
    tx = optax.adam(learning_rate)


    # Training

    key_init, key = jax.random.split(key)
    variables = linear.init(key_init, specimen)
    state = TrainState.create(
            apply_fn=linear.apply,
            params=variables['params'],
            tx=tx,
    )
    state = replicate(state)

    m_frequency = np.zeros((4,))
    loss_epoch_last = jnp.nan * jnp.ones(())
    pbar = trange(num_epochs)
    for epoch in pbar:
        loss_epoch = jnp.zeros(())
        iterator = iterator_jax(train_loader, device_count, image_processor, vit)
        for embedding, m in tqdm(iterator, leave=False):
            m_frequency += np.bincount(m.reshape(-1), minlength=4)
            state, loss = train_step(state, embedding, m)
            loss_epoch += loss.sum()

        ################ ################ ################ ################ ################ ################
        # break

        pbar.set_description(f"loss: {loss_epoch_last.item():.2f} --Epoch {epoch + 1}--> {loss_epoch.item():.2f}")
        loss_epoch_last = loss_epoch


    # Testing

    logit_bias_zero = replicate(jnp.zeros((4,)))
    hit_by_m = jnp.zeros((4,), dtype=jnp.int16)
    total_by_m = jnp.zeros((4,), dtype=jnp.int16)

    iterator = iterator_jax(test_loader, device_count, image_processor, vit)
    for embedding, m in tqdm(iterator, desc="Testing"):
        prediction = test_step(state, embedding, logit_bias_zero)
        weights = jnp.asarray(prediction == m, dtype=jnp.int16).reshape(-1)
        hit_by_m = hit_by_m + jnp.bincount(m.reshape(-1), weights, length=4)
        total_by_m = total_by_m + jnp.bincount(m.reshape(-1), length=4)

        ################ ################ ################ ################ ################ ################
        # break

    print(f"Hit: {hit_by_m}/{total_by_m}")


    # Logit Adjustment

    logit_bias = replicate(jnp.log(m_frequency))
    hit_by_m = jnp.zeros((4,), dtype=jnp.int16)
    total_by_m = jnp.zeros((4,), dtype=jnp.int16)

    iterator = iterator_jax(test_loader, device_count, image_processor, vit)
    for embedding, m in tqdm(iterator, desc="Logit Adjustment"):
        prediction = test_step(state, embedding, logit_bias)
        weights = jnp.asarray(prediction == m, dtype=jnp.int16).reshape(-1)
        hit_by_m = hit_by_m + jnp.bincount(m.reshape(-1), weights, length=4)
        total_by_m = total_by_m + jnp.bincount(m.reshape(-1), length=4)

        ################ ################ ################ ################ ################ ################
        # break

    print(f"Hit (Logit Adjustment): {hit_by_m}/{total_by_m}")


    # Embedding Adjustment

    kernel = unreplicate(state.params["kernel"])
    embedding_bias, residuals, rank, s = jnp.linalg.lstsq(kernel.T, unreplicate(logit_bias).T)

    key_init, key = jax.random.split(key)
    variables = linear.init(key_init, specimen)
    state = TrainState.create(
            apply_fn=linear.apply,
            params=variables['params'],
            tx=tx,
    )
    state = replicate(state)

    loss_epoch_last = jnp.nan * jnp.ones(())
    pbar = trange(num_epochs)
    for epoch in pbar:
        loss_epoch = jnp.zeros(())
        iterator = iterator_jax(train_loader, device_count, image_processor, vit)
        for embedding, m in tqdm(iterator, leave=False):
            embedding = embedding + 500 * embedding_bias
            state, loss = train_step(state, embedding, m)
            loss_epoch += loss.sum()

        ################ ################ ################ ################ ################ ################
        # break

        pbar.set_description(f"loss: {loss_epoch_last.item():.2f} --Epoch {epoch + 1}--> {loss_epoch.item():.2f}")
        loss_epoch_last = loss_epoch


    # Re-Testing

    hit_by_m = jnp.zeros((4,), dtype=jnp.int16)
    total_by_m = jnp.zeros((4,), dtype=jnp.int16)

    iterator = iterator_jax(test_loader, device_count, image_processor, vit)
    for embedding, m in tqdm(iterator, desc="Testing"):
        embedding = embedding + 500 * embedding_bias
        prediction = test_step(state, embedding, logit_bias_zero)
        weights = jnp.asarray(prediction == m, dtype=jnp.int16).reshape(-1)
        hit_by_m = hit_by_m + jnp.bincount(m.reshape(-1), weights, length=4)
        total_by_m = total_by_m + jnp.bincount(m.reshape(-1), length=4)

        ################ ################ ################ ################ ################ ################
        # break

    print(f"Hit (Embedding Adjustment): {hit_by_m}/{total_by_m}")


def iterator_jax(loader, device_count, image_processor, vit):
    for labeled_batch in loader:
        X, y, metadata = labeled_batch
        remainder = X.shape[0] % device_count
        if remainder > 0:
            X = X[:-remainder]
            y = y[:-remainder]
            metadata = metadata[:-remainder]

        N = X.shape[0] // device_count
        y = y.reshape((device_count, N))
        z = metadata[:, 0].reshape((device_count, N))
        m = jnp.array(y * 2 + z)

        inputs = image_processor(X, return_tensors="np")
        pixel_values = inputs["pixel_values"]
        pixel_values = pixel_values.reshape(device_count, N, *pixel_values.shape[1:])
        outputs = vit(pixel_values)
        # only look at the [CLS] token
        # FIXME: first or last?
        embedding = outputs.last_hidden_state[
                :, :, 0, :
        ]

        yield embedding, m


@partial(jax.pmap, axis_name='batch', donate_argnums=(0,))
def train_step(state: TrainState, embedding: jax.Array, m: jax.Array) -> Tuple[TrainState, jax.Array]:
    @jax.value_and_grad
    def loss_fn(params):
        variables = {'params': params}
        logits = state.apply_fn(variables, embedding)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, m)
        return loss.sum()

    loss, grads = loss_fn(state.params)
    grads = jax.lax.psum(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)

    return state, loss


@jax.pmap
def test_step(state: TrainState, embedding: jax.Array, logit_bias: jax.Array) -> jax.Array:
    variables = {'params': state.params}
    logits = state.apply_fn(variables, embedding)
    prediction = jnp.argmax(logits - logit_bias, axis=-1)

    return prediction


if __name__ == "__main__":
    num_epochs = 8
    batch_size = 64
    learning_rate = 1e-3
    main(num_epochs, batch_size, learning_rate)
