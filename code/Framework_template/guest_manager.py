import logging
from fedml_api.distributed.classical_vertical_fl.message_define import MyMessage
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager
import numpy as np


class GuestManager(ServerManager):
    def __init__(self, args, comm, rank, size, guest_trainer):
        super().__init__(args, comm, rank, size)

        self.guest_trainer = guest_trainer
        self.round_num = args.comm_round
        self.round_idx = 0
        self.inference = args.inference
        self.args = args
        self.guest_idx = 0
        self.host_idx = 0
        self.fuzz_round = 0

    def run(self):
        # if self.args.inference == False:
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id)
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_LOGITS,
                                              self.handle_message_receive_logits_from_client)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SALIENCY,
                                              self.handle_message_receive_saliency_from_client)

    def handle_message_receive_logits_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        host_train_logits = msg_params.get(MyMessage.MSG_ARG_KEY_TRAIN_LOGITS)
        host_test_logits = msg_params.get(MyMessage.MSG_ARG_KEY_TEST_LOGITS)
        self.guest_trainer.add_client_local_result(sender_id - 1, host_train_logits, host_test_logits)
        b_all_received = self.guest_trainer.check_whether_all_receive()

        if b_all_received and self.args.inference == False and self.args.fuzz == False:
            # logging.info("**********************************ROUND INDEX = " + str(self.round_idx))
            host_gradient = self.guest_trainer.train(self.round_idx)

            for receiver_id in range(1, self.size):
                self.send_message_to_client(receiver_id, host_gradient)

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num * self.guest_trainer.get_batch_num():
                self.finish()
        elif b_all_received and self.args.inference:
            self.guest_trainer.inference(self.round_idx)
            self.round_idx += 1
            if self.round_idx == 1:
                self.finish()
        elif b_all_received and self.args.fuzz:
            logging.info('Start fuzzing ......')
            # FIXME: fuzz1: compute the noised data and masked data, and their gradients and guest_saliency score. // DONE
            noised_guest_data, host_gradient1, masked_guest_data, host_gradient2 = self.guest_trainer.fuzz1(self.guest_idx, self.host_idx)
            self.guest_trainer.guest_masked_data = masked_guest_data
            logging.info('Masked Data computed, sending gradients ......')

            for receiver_id in range(1, self.size):
                message = Message(MyMessage.MSG_TYPE_S2C_GRADIENT, self.get_sender_id(), receiver_id)
                message.add_params(MyMessage.MSG_ARG_KEY_NOISED_GRADIENT, host_gradient1)
                message.add_params(MyMessage.MSG_ARG_KEY_MASKED_GRADIENT, host_gradient2)
                message.add_params(MyMessage.MSG_ARG_KEY_HOST_ID, self.host_idx)
                self.send_message(message)
                # self.send_message_to_client_host_id(receiver_id, self.host_idx)

    def handle_message_receive_saliency_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        host_noised_saliency = msg_params.get(MyMessage.MSG_ARG_KEY_NOISED_SALIENCY)
        host_masked_saliency = msg_params.get(MyMessage.MSG_ARG_KEY_MASKED_SALIENCY)
        # FIXME: add saliency socres to guest_trainer. // DONE
        logging.info('Saliency score received......')
        self.guest_trainer.add_client_saliency_result(sender_id - 1, host_noised_saliency, host_masked_saliency)
        b_all_received = self.guest_trainer.check_whether_all_receive()
        print(self.guest_trainer.host_local_noised_saliency)
        if b_all_received and self.args.fuzz:
            # FIXME: fuzz2: after receiving the saliency sent from host, compute the orig_ratio, orig_acc and new_ratio, new_acc.
            #               and update them in storage to perform fuzzing. // DONE
            flag = self.guest_trainer.fuzz2(self.guest_idx, self.host_idx)
            self.host_idx += 1
            # FIXME: choose next guest_idx
            if self.host_idx >= 10 or flag == True:
                self.fuzz_round += 1
                self.host_idx = 0
                self.guest_idx = np.random.choice(self.guest_trainer.S)
            # start the next and turn to host round
            noised_guest_data, host_gradient1, masked_guest_data, host_gradient2 = self.guest_trainer.fuzz1(self.guest_idx, self.host_idx)
            for receiver_id in range(1, self.size):
                message = Message(MyMessage.MSG_TYPE_S2C_GRADIENT, self.get_sender_id(), receiver_id)
                message.add_params(MyMessage.MSG_ARG_KEY_NOISED_GRADIENT, host_gradient1)
                message.add_params(MyMessage.MSG_ARG_KEY_MASKED_GRADIENT, host_gradient2)
                message.add_params(MyMessage.MSG_ARG_KEY_HOST_ID, self.host_idx)
                self.send_message(message)
                # self.send_message_to_client_host_id(receiver_id, self.host_idx)

            if self.fuzz_round >= 200 or self.guest_trainer.S.shape[0] <= 5:
                self.finish()
            

    def send_message_init_config(self, receive_id):
        print("Init message sent ......")
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        self.send_message(message)

    def send_message_to_client(self, receive_id, global_result):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADIENT, global_result)
        self.send_message(message)

    def send_message_to_client_host_id(self, receive_id, host_idx):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_HOST_ID, host_idx)
        self.send_message(message)
