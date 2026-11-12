
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# 1. LOAD INPUT IMAGE
# ==========================================

image_path = "input.jpg"

image = Image.open(image_path).convert("L")
img = np.array(image)

# Histogram
hist, _ = np.histogram(
    img.flatten(), bins=256, range=(0, 256)
)

hist = hist.astype(float)
prob = hist / (hist.sum() + 1e-12)

# Number of thresholds
N_THRESHOLDS = 3

# GWO parameters
N_WOLVES = 20
MAX_ITER = 100

# ==========================================
# 2. OBJECTIVE 1: OTSU FITNESS
# ==========================================

def otsu_fitness(thresholds):

    thresholds = np.sort(
        np.clip(np.rint(thresholds).astype(int), 1, 254)
    )

    bins = np.arange(256)
    total_mean = np.sum(bins * prob)

    boundaries = np.concatenate(
        ([0], thresholds, [256])
    )

    between_variance = 0.0

    for i in range(len(boundaries) - 1):

        lo = boundaries[i]
        hi = boundaries[i + 1]

        p = prob[lo:hi]
        weight = np.sum(p)

        if weight < 1e-12:
            continue

        mean = np.sum(
            bins[lo:hi] * p
        ) / weight

        between_variance += (
            weight * (mean - total_mean) ** 2
        )

    total_variance = np.sum(
        prob * (bins - total_mean) ** 2
    )

    # Normalize between-class variance
    score = between_variance / (
        total_variance + 1e-12
    )

    return -score


# ==========================================
# 3. OBJECTIVE 2: KAPUR ENTROPY
# ==========================================

def kapur_fitness(thresholds):

    thresholds = np.sort(
        np.clip(np.rint(thresholds).astype(int), 1, 254)
    )

    boundaries = np.concatenate(
        ([0], thresholds, [256])
    )

    total_entropy = 0.0

    for i in range(len(boundaries) - 1):

        lo = boundaries[i]
        hi = boundaries[i + 1]

        p = prob[lo:hi]
        weight = np.sum(p)

        if weight < 1e-12:
            continue

        conditional_prob = p / weight

        entropy = -np.sum(
            conditional_prob *
            np.log(conditional_prob + 1e-12)
        )

        total_entropy += entropy

    # Normalize by maximum possible entropy
    score = total_entropy / (
        len(boundaries) * np.log(256)
    )

    return -score


# ==========================================
# 4. GREY WOLF OPTIMIZER
# ==========================================

def GWO(objective):

    wolves = np.random.uniform(
        1, 255,
        (N_WOLVES, N_THRESHOLDS)
    )

    alpha_pos = np.zeros(N_THRESHOLDS)
    beta_pos = np.zeros(N_THRESHOLDS)
    delta_pos = np.zeros(N_THRESHOLDS)

    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")

    convergence = []

    for iteration in range(MAX_ITER):

        # Evaluate wolves
        for i in range(N_WOLVES):

            fitness = objective(wolves[i])

            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()

                beta_score = alpha_score
                beta_pos = alpha_pos.copy()

                alpha_score = fitness
                alpha_pos = wolves[i].copy()

            elif fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()

                beta_score = fitness
                beta_pos = wolves[i].copy()

            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = wolves[i].copy()

        # Update parameter a
        a = 2 - 2 * iteration / MAX_ITER

        # Update wolf positions
        for i in range(N_WOLVES):

            for leader in [
                alpha_pos, beta_pos, delta_pos
            ]:

                r1 = np.random.rand(N_THRESHOLDS)
                r2 = np.random.rand(N_THRESHOLDS)

                A = 2 * a * r1 - a
                C = 2 * r2

                D = np.abs(
                    C * leader - wolves[i]
                )

                X = leader - A * D

                if leader is alpha_pos:
                    X1 = X.copy()
                elif leader is beta_pos:
                    X2 = X.copy()
                else:
                    X3 = X.copy()

            wolves[i] = (
                X1 + X2 + X3
            ) / 3

            wolves[i] = np.clip(
                wolves[i], 1, 255
            )

        convergence.append(alpha_score)

    best_thresholds = np.sort(
        np.clip(
            np.rint(alpha_pos).astype(int),
            1, 254
        )
    )

    return best_thresholds, convergence


# ==========================================
# 5. IMAGE SEGMENTATION
# ==========================================

def segment_image(image, thresholds):

    segmented = np.zeros_like(image)

    for i, threshold in enumerate(thresholds):
        if i == 0:
            segmented[image <= threshold] = (
                255 * i // len(thresholds)
            )
        else:
            previous = thresholds[i - 1]

            segmented[
                (image > previous) &
                (image <= threshold)
            ] = 255 * i // len(thresholds)

    segmented[image > thresholds[-1]] = 255

    return segmented


# ==========================================
# 6. RUN OBJECTIVE 1: OTSU
# ==========================================

print("Running GWO with Otsu objective...")

otsu_thresholds, otsu_curve = GWO(
    otsu_fitness
)

otsu_result = segment_image(
    img, otsu_thresholds
)

print("Otsu thresholds:", otsu_thresholds)


# ==========================================
# 7. RUN OBJECTIVE 2: KAPUR
# ==========================================

print("Running GWO with Kapur objective...")

kapur_thresholds, kapur_curve = GWO(
    kapur_fitness
)

kapur_result = segment_image(
    img, kapur_thresholds
)

print("Kapur thresholds:", kapur_thresholds)


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

plt.figure(figsize=(14, 9))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(otsu_result, cmap="gray")
plt.title("GWO + Otsu Segmentation")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(kapur_result, cmap="gray")
plt.title("GWO + Kapur Segmentation")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.plot(
    np.arange(1, MAX_ITER + 1),
    -np.array(otsu_curve),
    label="Otsu"
)
plt.plot(
    np.arange(1, MAX_ITER + 1),
    -np.array(kapur_curve),
    label="Kapur"
)

plt.xlabel("Iterations")
plt.ylabel("Objective Score")
plt.title("GWO Convergence Curves")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
