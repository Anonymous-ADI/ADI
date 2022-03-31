from fedml_api.distributed.classical_vertical_fl.message_define import MyMessage
from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message


class HostManager(ClientManager):
    def __init__(self, args, comm, rank, size, trainer):
        super().__init__(args, comm, rank, size)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.args = args

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADIENT,
                                              self.handle_message_receive_gradient_from_server)

    def handle_message_init(self, msg_params):
        self.round_idx = 0
        if self.args.inference == False and self.args.fuzz == False:
            self.__train()
        else:
            self.__inference()

    def handle_message_receive_gradient_from_server(self, msg_params):
        if self.args.inference == False and self.args.fuzz == False:
            gradient = msg_params.get(MyMessage.MSG_ARG_KEY_GRADIENT)
            self.trainer.update_model(gradient, self.round_idx)
            self.round_idx += 1
            self.__train()
            if self.round_idx == self.num_rounds * self.trainer.get_batch_num():
                self.finish()
        elif self.args.inference == True:
            self.round_idx += 1
            self.__inference()
            if self.round_idx == 1:
                self.finish()
        elif self.args.fuzz == True:
            noised_gradient = msg_params.get(MyMessage.MSG_ARG_KEY_NOISED_GRADIENT)
            masked_gradient = msg_params.get(MyMessage.MSG_ARG_KEY_MASKED_GRADIENT)
            host_idx = msg_params.get(MyMessage.MSG_ARG_KEY_HOST_ID)
            # FIXME: calculate the saliency scores. // DONE
            noised_saliency = self.trainer.cal_saliency(noised_gradient, host_idx)
            masked_saliency = self.trainer.cal_saliency(masked_gradient, host_idx)
            message = Message(MyMessage.MSG_TYPE_C2S_SALIENCY, self.get_sender_id(), 0)
            message.add_params(MyMessage.MSG_ARG_KEY_NOISED_SALIENCY, noised_saliency)
            message.add_params(MyMessage.MSG_ARG_KEY_MASKED_SALIENCY, masked_saliency)
            self.send_message(message)


    def send_model_to_server(self, receive_id, host_train_logits, host_test_logits):
        message = Message(MyMessage.MSG_TYPE_C2S_LOGITS, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_TRAIN_LOGITS, host_train_logits)
        message.add_params(MyMessage.MSG_ARG_KEY_TEST_LOGITS, host_test_logits)
        self.send_message(message)

    def __train(self):
        host_train_logits, host_test_logits = self.trainer.computer_logits(self.round_idx)
        self.send_model_to_server(0, host_train_logits, host_test_logits)

    def __inference(self):
        print("Inference on Host start ......")
        host_train_logits, host_test_logits = self.trainer.computer_logits_inference(self.round_idx)
        self.send_model_to_server(0, host_train_logits, host_test_logits)
