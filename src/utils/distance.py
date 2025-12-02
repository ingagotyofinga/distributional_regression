from geomloss import SamplesLoss

def find_closest_mu(dataset, mu_0, top_k=1, blur=0.5):
    loss_fn = SamplesLoss("sinkhorn", p=2, blur=blur)
    distances = []

    for i in range(len(dataset)):
        mu_i, _ = dataset[i]
        dist = loss_fn(mu_0, mu_i).item()
        distances.append((i, dist))

    distances.sort(key=lambda x: x[1])
    return distances[:top_k]
