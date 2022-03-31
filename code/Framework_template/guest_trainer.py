import logging

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import optim, nn
import copy
from torch.autograd import Variable


class GuestTrainer(object):
    def __init__(self, client_num, device, X_train, y_train, X_test, y_test, model_feature_extractor, model_classifier,
                 args):
        self.client_num = client_num
        self.args = args
        self.device = device

        # training dataset
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = args.batch_size

        self.model_path = '/checkpoint/'

        N = self.X_train.shape[0]
        residual = N % args.batch_size
        if residual == 0:
            self.n_batches = N // args.batch_size
        else:
            self.n_batches = N // args.batch_size + 1
        self.batch_idx = 0
        logging.info("number of sample = %d" % N)
        logging.info("batch_size = %d" % self.batch_size)
        logging.info("number of batches = %d" % self.n_batches)

        # model
        self.model_feature_extractor = model_feature_extractor
        self.model_feature_extractor.to(device)
        self.optimizer_fe = optim.SGD(self.model_feature_extractor.parameters(), momentum=0.9, weight_decay=0.01,
                                      lr=self.args.lr)

        self.model_classifier = model_classifier
        self.model_classifier.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer_classifier = optim.SGD(self.model_classifier.parameters(), momentum=0.9, weight_decay=0.01,
                                              lr=self.args.lr)

        self.host_local_train_logits_list = dict()
        self.host_local_test_logits_list = dict()

        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.host_local_noised_saliency = dict()
        self.host_local_masked_saliency = dict()

        self.guest_local_noised_saliency = None
        self.guest_local_masked_saliency = None
        self.guest_masked_data = None

        self.loss_list = list()

        if args.inference:
            self.model_feature_extractor_inference = model_feature_extractor
            self.model_feature_extractor_inference.load(self.model_path + 'feature_guest_' + '47.pth')
            self.model_feature_extractor_inference.to(device)
            self.model_feature_extractor_inference.eval()

            self.model_classifier_inference = model_classifier
            self.model_classifier_inference.load(self.model_path + 'classifier_guest' + '47.pth')
            self.model_classifier_inference.to(device)
            self.model_classifier_inference.eval()

        if args.fuzz:
            self.model_feature_extractor_inference = model_feature_extractor
            self.model_feature_extractor_inference.load(self.model_path + 'feature_guest_' + '47.pth')
            self.model_feature_extractor_inference.to(device)
            self.model_feature_extractor_inference.eval()

            self.model_classifier_inference = model_classifier
            self.model_classifier_inference.load(self.model_path + 'classifier_guest' + '47.pth')
            self.model_classifier_inference.to(device)
            self.model_classifier_inference.eval()

            self.A = []
            Seed = np.arange(10)
            self.S = Seed
            self.Q = []
            self.Acc = []
            self.Y = []

    def get_batch_num(self):
        return self.n_batches

    def add_client_local_result(self, index, host_train_logits, host_test_logits):
        # logging.info("add_client_local_result. index = %d" % index)
        if self.args.fuzz == 0:
            self.host_local_train_logits_list[index] = host_train_logits
            self.host_local_test_logits_list[index] = host_test_logits
            self.flag_client_model_uploaded_dict[index] = True
        elif self.args.fuzz == 1:
            self.host_local_test_logits_list = host_test_logits
            self.flag_client_model_uploaded_dict[index] = True
            for id_guest in self.S:
                batch_x = self.X_test[id_guest: id_guest + 1]
                batch_y = self.y_test[id_guest: id_guest + 1]
                self.Q.append(batch_x)
                self.Y.append(batch_y)
                batch_x = torch.tensor(batch_x).float().to(self.device)
                self.Acc.append(self.advtest(batch_x, batch_y))

    def add_client_saliency_result(self, index, host_noised_saliency, host_masked_saliency):
        self.host_local_noised_saliency[index] = host_noised_saliency
        self.host_local_masked_saliency[index] = host_masked_saliency
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def train(self, round_idx):
        batch_x = self.X_train[self.batch_idx * self.batch_size: self.batch_idx * self.batch_size + self.batch_size]
        batch_y = self.y_train[self.batch_idx * self.batch_size: self.batch_idx * self.batch_size + self.batch_size]
        batch_x = torch.tensor(batch_x).float().to(self.device)
        batch_y = torch.tensor(batch_y).float().to(self.device)

        extracted_feature = self.model_feature_extractor.forward(batch_x)
        guest_logits = self.model_classifier.forward(extracted_feature)
        self.batch_idx += 1
        if self.batch_idx == self.n_batches:
            self.batch_idx = 0

        guest_logits = guest_logits.cpu().detach().numpy()
        for k in self.host_local_train_logits_list.keys():
            host_logits = self.host_local_train_logits_list[k]
            guest_logits += host_logits

        guest_logits = torch.tensor(guest_logits, requires_grad=True).float().to(self.device)
        batch_y = batch_y.type_as(guest_logits)

        # calculate the gradient until the logits for hosts
        class_loss = self.criterion(guest_logits, batch_y)
        grads = torch.autograd.grad(outputs=class_loss, inputs=guest_logits)

        loss = class_loss.item()
        self.loss_list.append(loss)

        # continue BP
        back_grad = self._bp_classifier(extracted_feature, grads)
        self._bp_feature_extractor(batch_x, back_grad)

        gradients_to_hosts = grads[0].cpu().detach().numpy()
        # logging.info("gradients_to_hosts = " + str(gradients_to_hosts))

        # for test and save model
        if (round_idx + 1) % self.args.frequency_of_the_test == 0:

            model_feature_path = self.model_path + 'feature_guest_' + str(round_idx) + '.pth'
            self.model_feature_extractor.save(model_feature_path)
            model_classifier_path = self.model_path + 'classifier_guest' + str(round_idx) + '.pth'
            self.model_classifier.save(model_classifier_path)

            self._test(round_idx)

        return gradients_to_hosts

    def _bp_classifier(self, x, grads):
        x = x.clone().detach().requires_grad_(True)
        output = self.model_classifier(x)
        output.backward(gradient=grads)
        x_grad = x.grad
        self.optimizer_classifier.step()
        self.optimizer_classifier.zero_grad()
        return x_grad

    def _bp_feature_extractor(self, x, grads):
        output = self.model_feature_extractor(x)
        output.backward(gradient=grads)
        self.optimizer_fe.step()
        self.optimizer_fe.zero_grad()

    def _bp_classifier_fuzz(self, x, grads):
        x = x.clone().detach().requires_grad_(True)
        output = self.model_classifier_inference(x)
        output.backward(gradient=grads)
        x_grad = x.grad
        return x_grad
    
    def _bp_feature_extractor_fuzz(self, x, grads):
        x = x.clone()
        x.requires_grad = True
        output = self.model_feature_extractor_inference(x)
        output.backward(gradient=grads)
        x_grad = x.grad
        return x_grad

    def _test(self, round_idx):
        X_test = torch.tensor(self.X_test).float().to(self.device)
        y_test = self.y_test

        extracted_feature = self.model_feature_extractor.forward(X_test)
        guest_logits = self.model_classifier.forward(extracted_feature)

        guest_logits = guest_logits.cpu().detach().numpy()
        for k in self.host_local_test_logits_list.keys():
            host_logits = self.host_local_test_logits_list[k]
            guest_logits += host_logits
        y_prob_preds = self._sigmoid(np.sum(guest_logits, axis=1))

        threshold = 0.5
        y_hat_lbls, statistics = self._compute_correct_prediction(y_targets=y_test,
                                                                  y_prob_preds=y_prob_preds,
                                                                  threshold=threshold)
        acc = accuracy_score(y_test, y_hat_lbls)
        auc = roc_auc_score(y_test, y_prob_preds)
        ave_loss = np.mean(self.loss_list)
        self.loss_list = list()
        logging.info(
            "--- round_idx: {%d}, loss: {%s}, acc: {%s}, auc: {%s}" % (round_idx, str(ave_loss), str(acc), str(auc)))
        logging.info(precision_recall_fscore_support(y_test, y_hat_lbls, average="macro", warn_for=tuple()))

    def inference(self, round_idx):
        X_test = torch.tensor(self.X_test).float().to(self.device)
        y_test = self.y_test

        extracted_feature = self.model_feature_extractor_inference.forward(X_test)
        guest_logits = self.model_classifier_inference.forward(extracted_feature)

        guest_logits = guest_logits.cpu().detach().numpy()
        for k in self.host_local_test_logits_list.keys():
            host_logits = self.host_local_test_logits_list[k]
            guest_logits += host_logits
        y_prob_preds = self._sigmoid(np.sum(guest_logits, axis=1))

        threshold = 0.5
        y_hat_lbls, statistics = self._compute_correct_prediction(y_targets=y_test,
                                                                  y_prob_preds=y_prob_preds,
                                                                  threshold=threshold)
        acc = accuracy_score(y_test, y_hat_lbls)
        auc = roc_auc_score(y_test, y_prob_preds)
        ave_loss = np.mean(self.loss_list)
        self.loss_list = list()
        logging.info(
            "--- round_idx: {%d}, loss: {%s}, acc: {%s}, auc: {%s}" % (round_idx, str(ave_loss), str(acc), str(auc)))
        logging.info(precision_recall_fscore_support(y_test, y_hat_lbls, average="macro", warn_for=tuple()))

    def before_fuzz(self, round_idx):
        batch_x = self.X_test[round_idx: round_idx + 1]
        batch_y = self.y_test[round_idx: round_idx + 1]
        batch_x = torch.tensor(batch_x).float().to(self.device)
        batch_y = torch.tensor(batch_y).float().to(self.device)
        extracted_feature = self.model_feature_extractor_inference.forward(batch_x)
        guest_logits = self.model_classifier_inference.forward(extracted_feature)
        for k in self.host_local_test_logits_list.keys():
            host_logits = self.host_local_test_logits_list[k]
            guest_logits += host_logits[round_idx: round_idx + 1]
        guest_logits = torch.tensor(guest_logits, requires_grad=True).float().to(self.device)
        batch_y = batch_y.type_as(guest_logits)
        class_loss = self.criterion(guest_logits, batch_y)
        grads = torch.autograd.grad(outputs=class_loss, inputs=guest_logits)
        gradients_to_hosts = grads[0].cpu().detach().numpy()

        return gradients_to_hosts

    def fuzz(self, round_idx):
        A = []
        Seed = np.arange(50)
        Q = []
        Acc = []
        for id_guest in Seed:
            batch_x = self.X_test[id_guest: id_guest + 1]
            batch_y = self.y_test[id_guest: id_guest + 1]
            Q.append(batch_x)
            batch_x = torch.tensor(batch_x).float().to(self.device)
            Acc.append(self.advtest(batch_x, batch_y))

        while len(A) < 10:
            id_guest = np.random.choice(Seed)
            batch_x = Q[id_guest]
            batch_y = self.y_test[id_guest: id_guest + 1]
            batch_x = torch.tensor(batch_x).float().to(self.device)
            # batch_y = torch.tensor(batch_y).float().to(self.device)

            batch_x.requires_grad = True
            for id_host in range(20):
                extracted_feature = self.model_feature_extractor_inference.forward(batch_x)
                guest_logits = self.model_classifier_inference.forward(extracted_feature)
                guest_logits = guest_logits.cpu().detach().numpy()
                
                for k in self.host_local_test_logits_list.keys():
                    host_logits = self.host_local_test_logits_list[k]
                    guest_logits += host_logits[id_host: id_host + 1]

                y_prob_preds = self._sigmoid(np.sum(guest_logits, axis=1))
                threshold = 0.5
                y_hat_lbls, statistics = self._compute_correct_prediction(y_targets=y_test,
                                                                        y_prob_preds=y_prob_preds,
                                                                        threshold=threshold)
                acc = accuracy_score(batch_y, y_hat_lbls)
                auc = roc_auc_score(batch_y, y_prob_preds)
                highest_score = self.smooth_grad(batch_x, guest_logits, batch_y)
                highest_noised_x = batch_x.clone()

                orig_mask = self.iccv17_mask(batch_x, id_guest)
                orig_acc = Acc[id_guest]

                for noise_index in range(50):
                    noised_inp_guest = batch_x + (torch.randn(batch_x.size())).to(device) * 0.05
                    noised_inp_guest.data.clamp_(-9, 97)
                    extracted_feature = self.model_feature_extractor_inference.forward(noised_inp_guest)
                    guest_logits_noise = self.model_classifier_inference.forward(extracted_feature)
                    guest_logits_noise = guest_logits_noise.cpu().detach().numpy()
                    for k in self.host_local_test_logits_list.keys():
                        host_logits = self.host_local_test_logits_list[k]
                        guest_logits_noise += host_logits[id_guest: id_guest + 1]
                    # guest_logits = torch.tensor(guest_logits, requires_grad=True).float().to(self.device)
                    temp_score = self.smooth_grad(noised_inp_guest, guest_logits_noise, batch_y)
                    if temp_score > highest_score:
                        highest_noised_x = noised_inp_guest.clone()
                        highest_score = temp_score

                noised_inp_guest = highest_noised_x
                new_mask = self.iccv17_mask(noised_inp_guest, id_host)

                if y_hat_lbls == batch_y:
                    mask = 1 + (1 - new_mask) * 0.2
                    tt_inp = noised_inp_guest.mul(mask)
                    tt_inp.data.clamp_(-9, 97)

                    new_acc = self.advtest(tt_inp, batch_y)

                    extracted_feature = self.model_feature_extractor_inference.forward(tt_inp)
                    guest_logits = self.model_classifier_inference.forward(extracted_feature)
                    guest_logits = guest_logits.cpu().detach().numpy()

                    for k in self.host_local_test_logits_list.keys():
                        host_logits = self.host_local_test_logits_list[k]
                        guest_logits += host_logits[id_guest: id_guest + 1]

                    temp_score = self.smooth_grad(tt_inp, guest_logits, batch_y)

                    if new_acc > orig_acc and temp_score >= highest_score:
                        batch_x = tt_inp
                        orig_mask = self.iccv17_mask(batch_x, id_host)
                        orig_acc = new_acc
                        highest_score = temp_score
                else:
                    mask = (1.0 - orig_mask.mul(1.0 - new_mask)) + (1 - orig_mask) * 0.3
                    tt_inp = noised_inp_guest.mul(mask)
                    tt_inp.data.clamp_(-9, 97)

                    new_acc = self.advtest(tt_inp, batch_y)

                    extracted_feature = self.model_feature_extractor_inference.forward(tt_inp)
                    guest_logits = self.model_classifier_inference.forward(extracted_feature)
                    guest_logits = guest_logits.cpu().detach().numpy()

                    for k in self.host_local_test_logits_list.keys():
                        host_logits = self.host_local_test_logits_list[k]
                        guest_logits += host_logits[id_guest: id_guest + 1]

                    temp_score = self.smooth_grad(tt_inp, guest_logits, batch_y)

                    if new_acc > orig_acc and temp_score >= highest_score:
                        batch_x = tt_inp
                        orig_mask = self.iccv17_mask(batch_x, id_host)
                        orig_acc = new_acc
                        highest_score = temp_score

            if orig_acc > 0.9:
                A.append(batch_x.cpu().detach().numpy())
                logging.info('======Found %dth with acc: %f======' % (len(A), orig_acc))
                S.remove(id_guest)
                Acc[id_guest] = orig_acc
                Q[id_guest] = batch_x.cpu().data.numpy()
            elif orig_acc > Acc[id_guest]:
                A.append(batch_x.cpu().detach().numpy())
                logging.info('======Guest %d with acc: %f======' % (guest_idx, orig_acc))
                Acc[id_guest] = orig_acc
                Q[id_guest] = batch_x.cpu().data.numpy()
            else:
                logging.info('======Guest %d did not increase acc======' % (guest_idx))

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _compute_correct_prediction(self, y_targets, y_prob_preds, threshold=0.5):
        y_hat_lbls = []
        pred_pos_count = 0
        pred_neg_count = 0
        correct_count = 0
        for y_prob, y_t in zip(y_prob_preds, y_targets):
            if y_prob <= threshold:
                pred_neg_count += 1
                y_hat_lbl = 0
            else:
                pred_pos_count += 1
                y_hat_lbl = 1
            y_hat_lbls.append(y_hat_lbl)
            if y_hat_lbl == y_t:
                correct_count += 1

        return np.array(y_hat_lbls), [pred_pos_count, pred_neg_count, correct_count]

    def smooth_grad(self, batch_x, guest_logits, batch_y):
        extracted_feature = self.model_feature_extractor_inference.forward(batch_x)

        guest_logits = torch.tensor(guest_logits, requires_grad=True).float().to(self.device)
        batch_y = batch_y.type_as(guest_logits)

        # calculate the gradient until the logits for hosts
        class_loss = self.criterion(guest_logits, batch_y)
        grads = torch.autograd.grad(outputs=class_loss, inputs=guest_logits)

        loss = class_loss.item()
        # self.loss_list.append(loss)

        # continue BP
        back_grad = self._bp_classifier_fuzz(extracted_feature, grads)
        x_grad = self._bp_feature_extractor_fuzz(batch_x, back_grad)
        x_grad = x_grad.data.cpu().numpy()

        smooth_grad = np.mean(x_grad * x_grad)
        return smooth_grad

    def iccv17_mask(self, img, id_host, batch_y):
        tv_beta = 3
        learning_rate = 0.01
        max_iterations = 1000
        l1_coeff = 0.1
        tv_coeff = 0.0002
        batch_y = torch.tensor(batch_y).float().to(self.device)
        img_numpy = copy.deepcopy(img.cpu().data.numpy())
        mask_init = np.ones(img_numpy.shape, dtype=np.float32)
        mask = self.numpy_to_torch(mask_init, requires_grad=True, device = self.device)
        optimizer = torch.optim.Adam([mask], lr=learning_rate)
        
        for i in range(max_iterations):
            extracted_feature = self.model_feature_extractor_inference.forward(img.mul(mask)).squeeze_(0).squeeze_(0)
            # logging.info('extracted_feature in iccv: ', extracted_feature)

            guest_logits = self.model_classifier_inference.forward(extracted_feature)
            guest_logits = guest_logits.cpu().detach().numpy()

            # for k in self.host_local_test_logits_list.keys():
            #     host_logits = self.host_local_test_logits_list[k]
            #     guest_logits += host_logits[id_host: id_host + 1]
            guest_logits += self.host_local_test_logits_list[id_host : id_host + 1]
            # logging.info('batch_y in iccv: ', batch_y)
            guest_logits = torch.tensor(guest_logits, requires_grad=True).float().to(self.device)
            # logging.info('guest_logits in iccv: ', guest_logits)
            class_loss = self.criterion(guest_logits, batch_y)
            grads = torch.autograd.grad(outputs=class_loss, inputs=guest_logits)
            loss = class_loss.item()

            back_grad = self._bp_classifier_fuzz(extracted_feature, grads)

            output_temp = self.model_feature_extractor_inference(img.mul(mask)).squeeze_(0).squeeze_(0)
            output_temp.backward(gradient=back_grad)
            optimizer.step()
            optimizer.zero_grad()
            mask.data.clamp_(0, 1)
        
        return mask

    def advtest(self, batch_x, batch_y):
        # print(len(self.host_local_test_logits_list))
        # print(batch_x.shape)
        extracted_feature = self.model_feature_extractor_inference.forward(batch_x)
        guest_logits = self.model_classifier_inference.forward(extracted_feature)
        guest_logits = guest_logits.cpu().detach().numpy()
        # print(guest_logits.shape)
        # print(self.host_local_test_logits_list.shape)
        # y_test = np.cat([batch_y[0]] * self.host_local_test_logits_list.shape[0])
        y_test = np.tile(batch_y, (self.host_local_test_logits_list.shape[0], 1))
        guest_logits = np.tile(guest_logits, (self.host_local_test_logits_list.shape[0], 1))
        # print(y_test)
        # print(guest_logits)
        # for k in self.host_local_test_logits_list.keys():
            # host_logits = self.host_local_test_logits_list[k]
        guest_logits += self.host_local_test_logits_list
        # print(guest_logits)
        y_prob_preds = self._sigmoid(np.sum(guest_logits, axis=1))
        # print(y_prob_preds)
        threshold = 0.5
        y_hat_lbls, statistics = self._compute_correct_prediction(y_targets=y_test,
                                                                  y_prob_preds=y_prob_preds,
                                                                  threshold=threshold)
        acc = accuracy_score(y_test, y_hat_lbls)
        logging.info("acc = %f" % acc)
        return acc

    # FIXME: fuzz1: compute the noised data and masked data, and their gradients and guest_saliency score.
    def fuzz1(self, id_guest, id_host):
        batch_x = self.Q[id_guest]
        batch_x = torch.tensor(batch_x).float().to(self.device)
        batch_y = self.Y[id_guest]
        batch_y = torch.tensor(batch_y).float().to(self.device)

        extracted_feature = self.model_feature_extractor_inference.forward(batch_x)
        guest_logits = self.model_classifier_inference.forward(extracted_feature)
        guest_logits = guest_logits.cpu().detach().numpy()
        
        # for k in self.host_local_test_logits_list.keys():
        #     host_logits = self.host_local_test_logits_list[k]
        #     guest_logits += host_logits[id_host: id_host + 1]
        guest_logits += self.host_local_test_logits_list[id_host : id_host + 1]
        y_prob_preds = self._sigmoid(np.sum(guest_logits, axis=1))
        threshold = 0.5
        y_hat_lbls, statistics = self._compute_correct_prediction(y_targets=batch_y.cpu().data.numpy(),
                                                                y_prob_preds=y_prob_preds,
                                                                threshold=threshold)
        # acc = accuracy_score(batch_y, y_hat_lbls)
        # auc = roc_auc_score(batch_y, y_prob_preds)
        highest_score = self.smooth_grad(batch_x, guest_logits, batch_y)
        highest_noised_x = batch_x.clone()
        logging.info("Temp saliency score: %f" % highest_score)
        orig_mask = self.iccv17_mask(batch_x, id_guest, batch_y)
        orig_acc = self.Acc[id_guest]

        for noise_index in range(50):
            noised_inp_guest = batch_x + (torch.randn(batch_x.size())).to(self.device) * 0.05
            noised_inp_guest.data.clamp_(-9, 97)
            extracted_feature = self.model_feature_extractor_inference.forward(noised_inp_guest)
            guest_logits_noise = self.model_classifier_inference.forward(extracted_feature)
            guest_logits_noise = guest_logits_noise.cpu().detach().numpy()
            # for k in self.host_local_test_logits_list.keys():
            #     host_logits = self.host_local_test_logits_list[k]
            #     guest_logits_noise += host_logits[id_guest: id_guest + 1]
            guest_logits_noise += self.host_local_test_logits_list[id_guest : id_guest + 1]
            # guest_logits = torch.tensor(guest_logits, requires_grad=True).float().to(self.device)
            temp_score = self.smooth_grad(noised_inp_guest, guest_logits_noise, batch_y)
            if temp_score > highest_score:
                highest_noised_x = noised_inp_guest.clone()
                highest_score = temp_score
        logging.info("Noised data computed ......")
        self.guest_local_noised_saliency = highest_score
        noised_inp_guest = highest_noised_x
        y_hat_lbls = torch.tensor(y_hat_lbls).float().to(self.device).unsqueeze_(0)
        print(y_hat_lbls, batch_y)
        new_mask = self.iccv17_mask(noised_inp_guest, id_host, y_hat_lbls)
        if y_hat_lbls == batch_y:
            mask = 1 + (1 - new_mask) * 0.2
            tt_inp = noised_inp_guest.mul(mask)
            tt_inp.data.clamp_(-9, 97)

        else:
            mask = (1.0 - orig_mask.mul(1.0 - new_mask)) + (1 - orig_mask) * 0.3
            tt_inp = noised_inp_guest.mul(mask)
            tt_inp.data.clamp_(-9, 97)
        tt_inp.squeeze_(0).squeeze_(0)
        extracted_feature = self.model_feature_extractor_inference.forward(noised_inp_guest)
        guest_logits = self.model_classifier_inference.forward(extracted_feature)
        guest_logits = guest_logits.cpu().detach().numpy()
        # for k in self.host_local_train_logits_list.keys():
        #     host_logits = self.host_local_train_logits_list[k]
        #     guest_logits += host_logits
        guest_logits += self.host_local_test_logits_list[id_host : id_host + 1]

        guest_logits = torch.tensor(guest_logits, requires_grad=True).float().to(self.device)
        batch_y = batch_y.type_as(guest_logits)

        # calculate the gradient until the logits for hosts
        class_loss = self.criterion(guest_logits, batch_y)
        grads = torch.autograd.grad(outputs=class_loss, inputs=guest_logits)

        host_gradient1 = grads[0].cpu().detach().numpy()

        tt_inp_clone = tt_inp.clone().detach().requires_grad_(True)
        extracted_feature = self.model_feature_extractor_inference.forward(tt_inp_clone)
        guest_logits = self.model_classifier_inference.forward(extracted_feature)
        guest_logits = guest_logits.cpu().detach().numpy()
        # for k in self.host_local_train_logits_list.keys():
        #     host_logits = self.host_local_train_logits_list[k]
        #     guest_logits += host_logits
        guest_logits += self.host_local_test_logits_list[id_host : id_host + 1]

        self.guest_local_masked_saliency = self.smooth_grad(tt_inp_clone.clone().detach(), guest_logits, batch_y)

        guest_logits = torch.tensor(guest_logits, requires_grad=True).float().to(self.device)
        # batch_y = batch_y.type_as(guest_logits)

        # calculate the gradient until the logits for hosts
        class_loss = self.criterion(guest_logits, batch_y)
        grads = torch.autograd.grad(outputs=class_loss, inputs=guest_logits)
        host_gradient2 = grads[0].cpu().detach().numpy()

        noised_guest_data = noised_inp_guest.cpu().data.numpy()
        masked_guest_data = tt_inp.cpu().data.numpy()
        logging.info("guest fuzz1 done ......")
        return noised_guest_data, host_gradient1, masked_guest_data, host_gradient2
    
    def fuzz2(self, guest_idx, host_idx):
        tt_inp = self.guest_masked_data
        tt_inp = self.numpy_to_torch(tt_inp, requires_grad=False, device=self.device).squeeze_(0).squeeze_(0)
        tt_inp.requires_grad_(True)
        logging.info("tt_inp size: " + str(tt_inp.shape))
        batch_y = self.Y[guest_idx]
        new_acc = self.advtest(tt_inp, batch_y)
        orig_acc = self.Acc[guest_idx]
        guest_score = self.guest_local_masked_saliency / self.guest_local_noised_saliency
        host_noised_saliency = 0.0
        host_masked_saliency = 0.0
        for k in self.host_local_noised_saliency.keys():
            host_noised_saliency += self.host_local_noised_saliency[k]
            host_masked_saliency += self.host_local_masked_saliency[k]
        host_score = host_masked_saliency / host_noised_saliency
        if new_acc > orig_acc and guest_score >= host_score:
            self.Acc[guest_idx] = new_acc
            self.Q[guest_idx] = self.guest_masked_data
        if new_acc > 0.9:
            self.A.append(self.guest_masked_data)
            self.S = np.delete(self.S, np.argwhere(self.S == guest_idx))
            return True
        return False

    def numpy_to_torch(self, img, requires_grad=True, device=0):
        if len(img.shape) < 3:
            output = np.float32([img])
        else:
            output = np.transpose(img, (2, 0, 1))

        output = torch.from_numpy(output)
        
        output = output.to(device)

        output.unsqueeze_(0)
        v = Variable(output, requires_grad = requires_grad)
        return v