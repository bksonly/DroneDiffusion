import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange

###################################################################
#################### Noise Predictor ##############################
###################################################################


class Mish(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jnp.tanh(jax.nn.softplus(x))


class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: [batch, ]
        returns: [batch, dim]
        """
        half_dim = self.dim // 2
        scale = jnp.log(10000) / (half_dim - 1)
        freqs = jnp.exp(-scale * jnp.arange(half_dim))  # [half_dim, ]
        emb = x[:, None] * freqs[None, :]  # [batch, half_dim]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)  # [batch, dim]
        return emb


class Downsample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: [batch, horizon, dim]
        returns: [batch, horizon // 2, dim]
        """
        conv = nn.Conv(
            features=self.dim, kernel_size=(3,), strides=(2,), padding="SAME"
        )
        return conv(x)


class Upsample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: [batch, horizon, dim]
        returns: [batch, horizon * 2, dim]
        """
        conv_transpose = nn.ConvTranspose(
            features=self.dim, kernel_size=(4,), strides=(2,), padding="SAME"
        )
        return conv_transpose(x)


class Conv1dBlock(nn.Module):
    out_channels: int
    kernel_size: int
    n_groups: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: [batch, horizon, inp_channels]
        returns: [batch, horizon, out_channels]
        """
        x = nn.Conv(
            features=self.out_channels, kernel_size=(self.kernel_size,), padding="SAME"
        )(x)
        x = nn.GroupNorm(num_groups=self.n_groups, epsilon=1e-5)(x)
        x = Mish()(x)
        return x


class Rearrange(nn.Module):
    pattern: str

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return rearrange(x, self.pattern)


class LayerNormCustom(nn.Module):
    dim: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: [batch, horizon, dim]
        returns: [batch, horizon, dim]
        """
        gamma = self.param("gamma", nn.initializers.ones, (1, 1, self.dim))
        beta = self.param("beta", nn.initializers.zeros, (1, 1, self.dim))
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(var + self.eps)
        out = normalized * gamma + beta
        return out


class PreNorm(nn.Module):
    dim: int
    fn: nn.Module

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = LayerNormCustom(dim=self.dim)(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    dim: int
    heads: int = 4
    dim_head: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: [batch, horizon, dim]
        returns: [batch, horizon, dim]
        """
        scale = self.dim_head**-0.5
        hidden_dim = self.dim_head * self.heads

        # 1. Project to Q, K, V
        qkv = nn.Conv(
            features=hidden_dim * 3, kernel_size=(1,), use_bias=False, name="to_qkv"
        )(
            x
        )  # Shape: [batch, horizon, 3*heads*dim_head]

        # 2. Split and reshape to [3, batch, heads, horizon, dim_head]
        q, k, v = rearrange(
            qkv,
            "b l (three h d) -> three b h l d",
            three=3,
            h=self.heads,
            d=self.dim_head,
        )

        # 3. Scale Queries
        q = q * scale

        # 4. Softmax keys over the horizon axis
        k = jax.nn.softmax(k, axis=2)

        # 5. Build context: [batch, heads, dim_head, dim_head]
        context = jnp.einsum("b h l d, b h l e -> b h d e", k, v)

        # 6. Attend: [batch, heads, horizon, dim_head]
        out = jnp.einsum("b h d e, b h l d -> b h l e", context, q)

        # 7. Merge heads and project out
        out = rearrange(
            out, "b h l d -> b l (h d)"
        )  # [batch, horizon, heads * dim_head]

        return nn.Conv(features=self.dim, kernel_size=(1,), name="to_out")(
            out
        )  # Shape: [batch, horizon, dim]


class ResidualTemporalBlock(nn.Module):
    inp_channels: int
    out_channels: int
    embed_dim: int
    kernel_size: int = 5

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        x: [batch, horizon, inp_channels]
        t: [batch, embed_dim]
        returns: [batch, horizon, out_channels]
        """

        # 1. Time embedding (Mish -> Dense(embed_dim -> out_channels))
        #    produces shape [batch, out_channels], then we add a length-axis
        t_emb = Mish()(t)
        t_emb = nn.Dense(features=self.out_channels, name="time_dense")(t_emb)
        t_emb = t_emb[:, None, :]  # Shape: [batch, 1, out_channels]

        # 2. Conv Block 1 + Time Embedding
        h = Conv1dBlock(
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            name="conv_block1",
        )(
            x
        )  # Shape: [batch, horizon, out_channels]
        h = h + t_emb  # Broadcast add over horizon

        # 3. Conv Block 2
        h = Conv1dBlock(
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            name="conv_block2",
        )(
            h
        )  # Shape: [batch, horizon, out_channels]

        # 4. Residual Connection
        if self.inp_channels != self.out_channels:
            res = nn.Conv(
                features=self.out_channels,
                kernel_size=(1,),
                padding="SAME",
                name="residual_conv",
            )(
                x
            )  # Shape: [batch, horizon, out_channels] (Project input to out_channels)
        else:
            res = x  # Identity connection if dimensions match

        return h + res  # Final output


class TemporalUnet(nn.Module):
    """
    A 1D U-Net with residual blocks, optional attention, and sinusoidal time conditioning.

    Args:
        horizon (int): Number of time steps per trajectory.
        transition_dim (int): Dimensionality of input per time step.
        cond_dim (int): Dimensionality of conditioning input.
        dim (int): Base channel width.
        dim_mults (tuple): Multipliers for each U-Net stage.
        attention (bool): If True, uses linear attention at each resolution.
    """

    horizon: int
    transition_dim: int
    cond_dim: int
    dim: int = 32
    dim_mults: tuple = (1, 2, 4, 8)
    attention: bool = False

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, cond: jnp.ndarray, time: jnp.ndarray
    ) -> jnp.ndarray:
        """
        x: [batch, horizon, transition_dim]
        cond: [batch, cond_dim]
        time: [batch, ]

        returns: [batch, horizon, transition_dim]
        """

        # Time + Condition Embedding
        t = SinusoidalPosEmb(dim=self.dim)(time)  # Shape: [batch, dim]
        t = nn.Dense(features=self.dim * 4, name="time_fc1")(t)
        t = Mish()(t)
        t = nn.Dense(features=self.dim, name="time_fc2")(t)  # Shape: [batch, dim]
        c = nn.Dense(features=self.dim, name="cond_fc")(cond)  # Shape: [batch, dim]

        t = t + c

        # Build Channel Pyramid
        dims = [self.transition_dim] + [self.dim * m for m in self.dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Encoder Path
        skips = []
        for idx, (dim_in, dim_out) in enumerate(in_out):
            x = ResidualTemporalBlock(
                inp_channels=dim_in,
                out_channels=dim_out,
                embed_dim=self.dim,
                kernel_size=5,
                name=f"down_res1_{idx}",
            )(x, t)
            x = ResidualTemporalBlock(
                inp_channels=dim_out,
                out_channels=dim_out,
                embed_dim=self.dim,
                kernel_size=5,
                name=f"down_res2_{idx}",
            )(x, t)

            if self.attention:
                x = PreNorm(dim=dim_out, fn=LinearAttention(dim=dim_out))(x)

            skips.append(x)
            if idx < len(in_out) - 1:
                x = Downsample1d(dim_out, name=f"downsample_{idx}")(x)

        # Bottleneck
        mid_c = dims[-1]
        x = ResidualTemporalBlock(
            inp_channels=mid_c,
            out_channels=mid_c,
            embed_dim=self.dim,
            kernel_size=5,
            name="bottleneck_res1",
        )(x, t)

        if self.attention:
            x_mid = PreNorm(dim=mid_c, fn=LinearAttention(dim=mid_c))(x_mid)

        x = ResidualTemporalBlock(
            inp_channels=mid_c,
            out_channels=mid_c,
            embed_dim=self.dim,
            kernel_size=5,
            name="bottleneck_res2",
        )(x, t)

        # Decoder Path
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # Pop the deepest skip first
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ResidualTemporalBlock(
                inp_channels=dim_out * 2,
                out_channels=dim_in,
                embed_dim=self.dim,
                kernel_size=5,
                name=f"up_res1_{idx}",
            )(x, t)
            x = ResidualTemporalBlock(
                inp_channels=dim_in,
                out_channels=dim_in,
                embed_dim=self.dim,
                kernel_size=5,
                name=f"up_res2_{idx}",
            )(x, t)

            if self.attention:
                x = PreNorm(dim=dim_in, fn=LinearAttention(dim=dim_in))(x)

            x = Upsample1d(dim_in, name=f"upsample_{idx}")(x)

        # Final Convolution
        x = Conv1dBlock(out_channels=self.dim, kernel_size=5, name="final_block")(x)
        x = nn.Conv(
            features=self.transition_dim,
            kernel_size=(1,),
            padding="SAME",
            name="final_conv",
        )(x)

        return x  # Shape: [batch, horizon, transition_dim]
