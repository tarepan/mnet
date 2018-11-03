import time
import torch

def convert1DInto2D(input1d):
    N_batch, channels, width = input1d.size()
    image2d = input1d.view(N_batch, 1, channels, width)
    return image2d

# https://github.com/pytorch/pytorch/issues/12901
# elementwise_mean_loss = lambda x, x_out: torch.mean(torch.abs(x-x_out.view_as(x)))

def train(args, net_G_A2B, net_G_B2A, net_D_A, net_D_B, device, train_loader_A, train_loader_B, opt_G, opt_D_A, opt_D_B, epoch, writer):
    """
    Train models only single epoch
    """
    # print
    start = time.time()
    print(f"\nEpoch {epoch} training start.")
    N_iter = 0
    # training mode
    net_G_A2B.train()
    net_G_B2A.train()
    net_D_A.train()
    net_D_B.train()
    # Lossess
    criterion_GAN = torch.nn.BCELoss(reduction='mean') # NS-GAN. my original.
    # criterion_cycle = elementwise_mean_loss # torch.nn.L1Loss(reduction='elementwise_mean')
    criterion_cycle = torch.nn.L1Loss(reduction='mean')
    criterion_identity = torch.nn.L1Loss(reduction='mean')
    # configs
    lambda_cyc = 10.0
    lambda_id = 5.0

    # lambda_id degradation based on epoch
    if epoch > 10000/81:
        lambda_id = 0

    # prepare recording
    ave_Ladv_A2B = 0
    ave_Ladv_B2A = 0
    ave_Ladv_D_A = 0
    ave_Ladv_D_B = 0
    ave_Lcyc_A = 0
    ave_Lcyc_B = 0
    ave_Lid_A = 0
    ave_Lid_B = 0
    # training cycle
    for batch_idx, (batch_A, batch_B) in enumerate(zip(train_loader_A, train_loader_B)):
        if type(args.batch_num_limit_train) is not int or batch_idx <= args.batch_num_limit_train:
            N_iter = N_iter + 1
            # make all 0/1 tensor for loss calculation
            # batch size should be same
            t = time.time()
            batch_size = batch_A.size()[0]
            zeros, ones = torch.zeros(batch_size, 1, device=device), torch.ones(batch_size, 1, device=device)
            # send to GPU
            real_A, real_B = batch_A.to(device), batch_B.to(device)
            #########################################
            ## Generator A2B & B2A
            # initialize
            opt_G.zero_grad()
            # Adversarial loss
            fake_B = net_G_A2B(real_A)
            # non-saturating GAN
            Lgen_A2B = criterion_GAN(net_D_B(convert1DInto2D(fake_B)), ones)
            fake_A = net_G_B2A(real_B)
            Lgen_B2A = criterion_GAN(net_D_A(convert1DInto2D(fake_A)), ones)
            # Cycle consistency loss
            # t = net_G_B2A(fake_B)
            # Lcyc_A2B2A = criterion_cycle(t, real_A)
            Lcyc_A2B2A = criterion_cycle(net_G_B2A(fake_B), real_A)
            Lcyc_B2A2B = criterion_cycle(net_G_A2B(fake_A), real_B)
            # Identity mapping loss
            Lid_B = criterion_identity(net_G_A2B(real_B), real_B)
            Lid_A = criterion_identity(net_G_B2A(real_A), real_A)
            # Total loss
            # print(f"\nfakeB: {fake_B.size()}")
            # print(fake_B)
            # print(f"\ncycleGenA: {t.size()}")
            # print(t)
            # print("diff")
            # print(t-fake_B)
            # print(f"Lgen_A2B: {Lgen_A2B.size()}")
            # print(Lgen_A2B)
            # print(f"Lcyc_A2B2A: {Lcyc_A2B2A.size()}")
            # print(Lcyc_A2B2A)
            Ltotal_G = Lgen_A2B + Lgen_B2A + lambda_cyc*Lcyc_A2B2A + lambda_cyc*Lcyc_B2A2B + lambda_id*Lid_B + lambda_id*Lid_A
            # backprop & update weight
            Ltotal_G.backward()
            opt_G.step()
            # Record losses
            ave_Ladv_A2B += Lgen_A2B.item()
            ave_Ladv_B2A += Lgen_B2A.item()
            ave_Lcyc_A += Lcyc_A2B2A.item()
            ave_Lcyc_B += Lcyc_B2A2B.item()
            ave_Lid_A += Lid_A.item()
            ave_Lid_B += Lid_B.item()
            #########################################
            ## Discriminator A
            start_D = time.time()
            opt_D_A.zero_grad()
            # calculate losses
            Lreal_A = criterion_GAN(net_D_A(convert1DInto2D(real_A.detach())), ones)
            Lfake_A = criterion_GAN(net_D_A(convert1DInto2D(fake_A.detach())), zeros)
            Ltotal_D_A = Lreal_A*0.5 + Lfake_A*0.5
            # backprop & update weight
            Ltotal_D_A.backward()
            opt_D_A.step()
            # Record loss
            ave_Ladv_D_A += Ltotal_D_A.item()
            #########################################
            ## Discriminator B
            opt_D_B.zero_grad()
            # calculate losses
            Lreal_B = criterion_GAN(net_D_B(convert1DInto2D(real_B.detach())), ones)
            Lfake_B = criterion_GAN(net_D_B(convert1DInto2D(fake_B.detach())), zeros)
            Ltotal_D_B = Lreal_B*0.5 + Lfake_B*0.5
            # backprop & update weight
            Ltotal_D_B.backward()
            opt_D_B.step()
            # Record loss
            ave_Ladv_D_B += Ltotal_D_B.item()
            #########################################
    # # record (number: iteration)
    batch_num = len(train_loader_A.dataset)/args.batch_size
    writer.add_scalar("L_adv/G_A2B",ave_Ladv_A2B/batch_num, epoch)
    writer.add_scalar("L_adv/G_B2A",ave_Ladv_B2A/batch_num, epoch)
    writer.add_scalar("L_adv/D_A",ave_Ladv_D_A/batch_num, epoch)
    writer.add_scalar("L_adv/D_B",ave_Ladv_D_B/batch_num, epoch)
    writer.add_scalar("L_cyc/cyc_A",ave_Lcyc_A/batch_num, epoch)
    writer.add_scalar("L_cyc/cyc_B",ave_Lcyc_B/batch_num, epoch)
    writer.add_scalar("L_id/id_A",ave_Lid_A/batch_num, epoch)
    writer.add_scalar("L_id/id_B",ave_Lid_B/batch_num, epoch)

    # (Additional) log setting
    print (f"epoch {epoch} training finished. {time.time() - start} [sec]")

        # if False:
        # # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
