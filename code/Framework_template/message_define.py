
class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_GRADIENT = 2

    # client to server
    MSG_TYPE_C2S_LOGITS = 3
    MSG_TYPE_C2S_SALIENCY = 4

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_TRAIN_LOGITS = "train_logits"
    MSG_ARG_KEY_TEST_LOGITS = "test_logits"
    MSG_ARG_KEY_GRADIENT = "gradient"
    MSG_ARG_KEY_NOISED_GRADIENT = "noised_gradient"
    MSG_ARG_KEY_MASKED_GRADIENT = "masked_gradient"
    MSG_ARG_KEY_HOST_ID = "host_id"
    MSG_ARG_KEY_NOISED_SALIENCY = "host_noised_saliency_score"
    MSG_ARG_KEY_MASKED_SALIENCY = "host_masked_saliency_score"