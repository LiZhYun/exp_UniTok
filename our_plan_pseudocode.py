class VAEModel:

    def __init__():
        self.quantizer = quantizer

    def forward():
        ...


class Quantizer:

    def __init__():
        self.register_buffer("percent", torch.tensor(1.0, dtype=torch.float), persistent=False)

    def forward(self, feature):
        """
        - feature: (b,h*w,c) float
        - self.codebook.weight: (n,c) float
        """
        quant, quant_err = match_code_with_feature_and_select(
            feature, self.codebook)  # (b,h*w,c) (b,h*w)

        quant = straight_through_grad_estimation(quant, feature)

        if self.training:
            thresh = calculate_threshold_that_will_enable_x_percent_of_quant(
                quant_err, self.percent)
            quant2 = quant.where(quant_err[:, :, None] < thresh, feature)  # (b,h*w,c)

        quant_loss = MSE(quant, feature.detach()) + beta * MSE(quant2.detach(), feature)
        return quant2


def train_epoch(model, dataset, optimiz, global_step):
    """
    - schedule_of_percent_values: (n_total_steps,) float, ranging from 0 to 1
    """
    for batch in dataset:
        # calculate percent and update model.quant.percent
        percent = schedule_of_percent_values[global_step]
        model.quant.percent.data[...] = percent
        # common stuff
        output = model(batch)
        loss = ...
        optimiz.step()
        global_step += 1
    return


def val_epoch():
    return
