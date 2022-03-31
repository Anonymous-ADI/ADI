import torch
from torch import nn, optim
import numpy as np


class HostTrainer(object):
    def __init__(self, client_index, device, X_train, X_test, model_feature_extractor, model_classifier, args):
        # device information
        self.client_index = client_index
        self.device = device
        self.args = args

        # training dataset
        self.X_train = X_train
        self.X_test = X_test
        self.batch_size = args.batch_size

        N = self.X_train.shape[0]
        residual = N % args.batch_size
        if residual == 0:
            self.n_batches = N // args.batch_size
        else:
            self.n_batches = N // args.batch_size + 1
        # logging.info("n_batches = %d" % self.n_batches)
        self.batch_idx = 0

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

        self.cached_extracted_features = None
        self.model_path = '/checkpoint/'

        if args.inference:
            self.model_feature_extractor_inference = model_feature_extractor
            self.model_feature_extractor_inference.load(self.model_path + 'feature_host_' + '47.pth')
            self.model_feature_extractor_inference.to(device)
            self.model_feature_extractor_inference.eval()

            self.model_classifier_inference = model_classifier
            self.model_classifier_inference.load(self.model_path + 'classifier_host' + '47.pth')
            self.model_classifier_inference.to(device)
            self.model_classifier_inference.eval()

        if args.fuzz:
            self.model_feature_extractor_inference = model_feature_extractor
            self.model_feature_extractor_inference.load(self.model_path + 'feature_host_' + '47.pth')
            self.model_feature_extractor_inference.to(device)
            self.model_feature_extractor_inference.eval()

            self.model_classifier_inference = model_classifier
            self.model_classifier_inference.load(self.model_path + 'classifier_host' + '47.pth')
            self.model_classifier_inference.to(device)
            self.model_classifier_inference.eval()

    def get_batch_num(self):
        return self.n_batches

    def computer_logits(self, round_idx):
        batch_x = self.X_train[self.batch_idx * self.batch_size: self.batch_idx * self.batch_size + self.batch_size]
        self.batch_x = torch.tensor(batch_x).float().to(self.device)
        self.extracted_feature = self.model_feature_extractor.forward(self.batch_x)
        logits = self.model_classifier.forward(self.extracted_feature)
        # copy to CPU host memory
        logits_train = logits.cpu().detach().numpy()
        self.batch_idx += 1
        if self.batch_idx == self.n_batches:
            self.batch_idx = 0

        # for test
        if (round_idx + 1) % self.args.frequency_of_the_test == 0:
            X_test = torch.tensor(self.X_test).float().to(self.device)
            extracted_feature = self.model_feature_extractor.forward(X_test)
            logits_test = self.model_classifier.forward(extracted_feature)
            logits_test = logits_test.cpu().detach().numpy()
        else:
            logits_test = None

        return logits_train, logits_test

    def computer_logits_inference(self, round_idx):
        # for test
        # if (round_idx + 1) % self.args.frequency_of_the_test == 0:
        print("Computing host logits......")
        X_test = torch.tensor(self.X_test[0:10]).float().to(self.device)
        extracted_feature = self.model_feature_extractor_inference.forward(X_test)
        logits_test = self.model_classifier_inference.forward(extracted_feature)
        logits_test = logits_test.cpu().detach().numpy()
        # else:
        #     logits_test = None

        return None, logits_test

    def update_model(self, gradient, round_idx):
        # logging.info("#######################gradient = " + str(gradient))
        gradient = torch.tensor(gradient).float().to(self.device)
        back_grad = self._bp_classifier(self.extracted_feature, gradient)
        self._bp_feature_extractor(self.batch_x, back_grad)
        if (round_idx + 1) % self.args.frequency_of_the_test == 0:
            model_feature_path = self.model_path + 'feature_host_' + str(round_idx) + '.pth'
            self.model_feature_extractor.save(model_feature_path)
            model_classifier_path = self.model_path + 'classifier_host' + str(round_idx) + '.pth'
            self.model_classifier.save(model_classifier_path)

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
        x = x.clone().detach().requires_grad_(True)
        output = self.model_feature_extractor_inference(x)
        output.backward(gradient=grads)
        x_grad = x.grad
        return x_grad

    def cal_saliency(self, gradient, host_idx):
        gradient = torch.tensor(gradient).float().to(self.device)
        batch_x = self.X_train[host_idx : host_idx + 1]
        batch_x = torch.tensor(batch_x).float().to(self.device)
        extracted_feature = self.model_feature_extractor.forward(batch_x)
        saliency = self.smooth_grad(batch_x, extracted_feature, gradient)
        return saliency

    def smooth_grad(self, batch_x, extracted_feature, grads):
        extracted_feature = self.model_feature_extractor_inference.forward(batch_x)
        # self.loss_list.append(loss)
        # continue BP
        back_grad = self._bp_classifier_fuzz(extracted_feature, grads)
        x_grad = self._bp_feature_extractor_fuzz(batch_x, back_grad)
        x_grad = x_grad.data.cpu().numpy()

        smooth_grad = np.mean(x_grad * x_grad)
        return smooth_grad