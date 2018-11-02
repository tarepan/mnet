import torch.nn.functional as F

def train(args, model, device, train_loader, optimizer, epoch, writer):
    # training mode
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # send to GPU
        data, target = data.to(device), target.to(device)
        # initialize
        optimizer.zero_grad()
        # inference
        output = model(data)
        # calculate loss
        loss = F.nll_loss(output, target)
        # backprop & update weight
        loss.backward()
        optimizer.step()

        # record (number: iteration)
        writer.add_scalar("data/loss",loss.item(), batch_idx + (epoch-1)*len(train_loader.dataset)/args.batch_size)

        # (Additional) log setting
#         if False:
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
