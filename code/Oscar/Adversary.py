import os
import io
import numpy as np
import cv2
import torch
import json
import torch.nn as nn
import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image

class AdvImageToFeat(object):

    def __init__(self, data_path, yaml_path, weights_path, num_obj=36):

        self.vg_classes = []
        with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
            for object in f.readlines():
                self.vg_classes.append(object.split(',')[0].lower().strip())
                
        self.vg_attrs = []
        with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
            for object in f.readlines():
                self.vg_attrs.append(object.split(',')[0].lower().strip())

        self.cfg = get_cfg()
        self.cfg.merge_from_file(yaml_path)
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
        self.cfg.MODEL.WEIGHTS = weights_path
        
        self.predictor = DefaultPredictor(self.cfg)

        self.NUM_OBJECTS = num_obj
        self.FEATURE_DIM = 2048

        self.mutated_fearure = None
        self.mutated_image = None
        self.original_image = None
        self.image_qq = None

    def set_image(self, image):
        self.mutated_image = image
        self.mutated_image.requires_grad = True

    def set_feature(self, feature):
        self.mutated_fearure = feature
        #self.mutated_fearure.requires_grad = True

    def set_mask(self, mask):
        self.mutated_mask = mask
        self.mutated_mask.requires_grad = True

    def zero_grad(self):
        self.predictor.model.zero_grad()

    def doit(self, raw_image, at_first=True, mask=None):
        raw_height, raw_width = raw_image.shape[:2]        
        if at_first:
            image = self.predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        else:
            image = raw_image
        self.predictor.model.zero_grad()

        self.original_image = image
        if not (mask is None):
            self.original_mask = mask
        image_height, image_width = image.shape[:2]
        self.image_height = image_height
        self.image_width = image_width
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)) / 255.0
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = self.predictor.model.preprocess_image(inputs)
        
        self.set_image(images.tensor)

        if not (mask is None):
            self.set_mask(
                torch.as_tensor(mask.astype("float32").transpose(2, 0, 1)).unsqueeze(0).cuda()
                )
        features = self.predictor.model.backbone((self.mutated_image if mask is None else self.mutated_image * self.mutated_mask))
        proposals, _ = self.predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in self.predictor.model.roi_heads.in_features]
        box_features = self.predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = self.predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            self.predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]
        
        box_width = boxes[:, 2] - boxes[:, 0]
        box_height = boxes[:, 3] - boxes[:, 1]
        scaled_width = box_width / image_width
        scaled_height = box_height / image_height
        scaled_x = boxes[:, 0] / image_width
        scaled_y = boxes[:, 1] / image_height
        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]
        spatial_features = torch.cat( (scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height), 1)
        oscar_features = torch.cat((feature_pooled, spatial_features), 1)
        
        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)
        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:], 
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=self.NUM_OBJECTS
            )
            if len(ids) >= self.NUM_OBJECTS:
                break
        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids]
        oscar_features = oscar_features[ids].detach()
        max_attr_prob = max_attr_prob[ids].detach()
        max_attr_label = max_attr_label[ids].detach()
        instances.attr_scores = max_attr_prob
        instances.attr_classes = max_attr_label
        self.set_feature(roi_features)

        if mask is None:
            return instances, roi_features, oscar_features
        else:
            return instances, roi_features, oscar_features, self.mutated_mask

    def doit_funny(self, raw_image, at_first=True, mask=None):
        raw_height, raw_width = raw_image.shape[:2]        
        # Preprocessing
        if at_first:
            image = self.predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        else:
            image = raw_image
        self.predictor.model.zero_grad()

        self.original_image = image
        if not (mask is None):
            self.original_mask = mask
        image_height, image_width = image.shape[:2]
        self.image_height = image_height
        self.image_width = image_width
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)) / 255.0
        self.image_qq = image
        self.image_qq.requires_grad = True
        inputs = [{"image": self.image_qq, "height": raw_height, "width": raw_width}]
        images = self.predictor.model.preprocess_image(inputs)

        if not (mask is None):
            self.set_mask(
                torch.as_tensor(mask.astype("float32").transpose(2, 0, 1)).unsqueeze(0).cuda()
                )
        features = self.predictor.model.backbone((images.tensor if mask is None else self.mutated_image * images.tensor))

        proposals, _ = self.predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]

        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in self.predictor.model.roi_heads.in_features]

        box_features = self.predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )

        feature_pooled = box_features.mean(dim=[2, 3])

        pred_class_logits, pred_attr_logits, pred_proposal_deltas = self.predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            self.predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]
        
        box_width = boxes[:, 2] - boxes[:, 0]
        box_height = boxes[:, 3] - boxes[:, 1]
        scaled_width = box_width / image_width
        scaled_height = box_height / image_height
        scaled_x = boxes[:, 0] / image_width
        scaled_y = boxes[:, 1] / image_height
        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]
        spatial_features = torch.cat( (scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height), 1)
        oscar_features = torch.cat((feature_pooled, spatial_features), 1)
        
        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)
        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:], 
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=self.NUM_OBJECTS
            )
            if len(ids) >= self.NUM_OBJECTS:
                break
        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids]#.detach()
        oscar_features = oscar_features[ids].detach()
        max_attr_prob = max_attr_prob[ids].detach()
        max_attr_label = max_attr_label[ids].detach()
        instances.attr_scores = max_attr_prob
        instances.attr_classes = max_attr_label

        self.set_feature(roi_features)

        if mask is None:
            return instances, roi_features, oscar_features
        else:
            return instances, roi_features, oscar_features, self.mutated_mask

    def doit_funny2(self, raw_image, at_first=True, mask=None):
        raw_height, raw_width = raw_image.shape[:2]
        if at_first:
            image = self.predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        else:
            image = raw_image
        self.predictor.model.zero_grad()

        self.original_image = image
        if not (mask is None):
            self.original_mask = mask
        image_height, image_width = image.shape[:2]
        self.image_height = image_height
        self.image_width = image_width
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)) / 255.0
        
        inputs = [{"image": self.image_qq, "height": raw_height, "width": raw_width}]
        images = self.predictor.model.preprocess_image(inputs)
        if not (mask is None):
            self.set_mask(
                torch.as_tensor(mask.astype("float32").transpose(2, 0, 1)).unsqueeze(0).cuda()
                )

        features = self.predictor.model.backbone((images.tensor if mask is None else images.tensor * self.mutated_mask))
        proposals, _ = self.predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]

        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in self.predictor.model.roi_heads.in_features]

        box_features = self.predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = self.predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            self.predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]
        
        box_width = boxes[:, 2] - boxes[:, 0]
        box_height = boxes[:, 3] - boxes[:, 1]
        scaled_width = box_width / self.image_width
        scaled_height = box_height / self.image_height
        scaled_x = boxes[:, 0] / self.image_width
        scaled_y = boxes[:, 1] / self.image_height
        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]
        spatial_features = torch.cat( (scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height), 1)
        oscar_features = torch.cat((feature_pooled, spatial_features), 1)
        
        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)
        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:], 
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=self.NUM_OBJECTS
            )
            if len(ids) >= self.NUM_OBJECTS:
                break
        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids]
        oscar_features = oscar_features[ids]
        max_attr_prob = max_attr_prob[ids].detach()
        max_attr_label = max_attr_label[ids].detach()
        instances.attr_scores = max_attr_prob
        instances.attr_classes = max_attr_label

        self.set_feature(roi_features)

        if mask is None:
            return instances, roi_features, oscar_features
        else:
            return instances, roi_features, oscar_features, self.mutated_mask

    def doit_infer(self, raw_image, at_first=True):
        with torch.no_grad():
            raw_height, raw_width = raw_image.shape[:2]
            if at_first:
                image = self.predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
            else:
                image = raw_image
            image_height, image_width = image.shape[:2]
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": raw_height, "width": raw_width}]
            images = self.predictor.model.preprocess_image(inputs)
            features = self.predictor.model.backbone(images.tensor)

            proposals, _ = self.predictor.model.proposal_generator(images, features, None)
            proposal = proposals[0]
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in self.predictor.model.roi_heads.in_features]
            box_features = self.predictor.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])
            pred_class_logits, pred_attr_logits, pred_proposal_deltas = self.predictor.model.roi_heads.box_predictor(feature_pooled)
            outputs = FastRCNNOutputs(
                self.predictor.model.roi_heads.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.predictor.model.roi_heads.smooth_l1_beta,
            )
            probs = outputs.predict_probs()[0]
            boxes = outputs.predict_boxes()[0]
            
            box_width = boxes[:, 2] - boxes[:, 0]
            box_height = boxes[:, 3] - boxes[:, 1]
            scaled_width = box_width / image_width
            scaled_height = box_height / image_height
            scaled_x = boxes[:, 0] / image_width
            scaled_y = boxes[:, 1] / image_height
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]
            spatial_features = torch.cat( (scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height), 1)
            oscar_features = torch.cat((feature_pooled, spatial_features), 1)
            
            attr_prob = pred_attr_logits[..., :-1].softmax(-1)
            max_attr_prob, max_attr_label = attr_prob.max(-1)
            
            # NMS
            for nms_thresh in np.arange(0.5, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes, probs, image.shape[1:], 
                    score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=self.NUM_OBJECTS
                )
                if len(ids) >= self.NUM_OBJECTS:
                    break
            instances = detector_postprocess(instances, raw_height, raw_width)
            roi_features = feature_pooled[ids].detach()
            oscar_features = oscar_features[ids].detach()
            max_attr_prob = max_attr_prob[ids].detach()
            max_attr_label = max_attr_label[ids].detach()
            instances.attr_scores = max_attr_prob
            instances.attr_classes = max_attr_label
            
            return instances, roi_features, oscar_features

    def mutate_from_grad(self, grad, lr=0.01):
        assert self.mutated_fearure.requires_grad == True
        grad = grad.cuda()
        if self.mutated_fearure is None or self.mutated_image is None:
            print('Image or Feature is None!')
            return None
        self.mutated_fearure.backward(grad[:, :self.FEATURE_DIM])
        gradients = self.mutated_image.grad[0].transpose(0, 1).transpose(1, 2).detach().cpu().numpy()
        assert gradients.shape == self.original_image.shape
        return (self.original_image - lr*gradients)

    def mutate_mask(self, grad, lr=0.01):
        assert self.mutated_fearure.requires_grad == True
        grad = grad.cuda()
        if self.mutated_fearure is None or self.mutated_mask is None:
            print('Mask or Feature is None!')
            return None
        self.mutated_fearure.backward(grad[:, :self.FEATURE_DIM])
        gradients = self.mutated_mask.grad[0].transpose(0, 1).transpose(1, 2).detach().cpu().numpy()
        assert gradients.shape == self.original_mask.shape
        return (self.original_mask - lr*gradients)

    def from_feat_to_img(self, raw_image, target_feats, local_epochs=50):
        dis_loss = nn.MSELoss()
        _, _, oscar_feat = self.doit_funny(raw_image, at_first=True, mask=None)
        dis_optimizer = torch.optim.Adam([self.image_qq], lr=10)
        for i in range(local_epochs):
            dis_optimizer.zero_grad()
            _, _, oscar_feat = self.doit_funny2(raw_image, at_first=True, mask=None)
            if oscar_feat.shape != target_feats.shape:
                min_shape = min([oscar_feat.shape[0], target_feats.shape[0]])
                loss = dis_loss(oscar_feat[:min_shape,:], target_feats[:min_shape,:])
            else:
                loss = dis_loss(oscar_feat, target_feats)
            print(loss)
            loss.backward(retain_graph=True)
            dis_optimizer.step()
            # self.image_qq.clamp_(0, 255)
            # self.image_qq = torch.clamp(self.image_qq, 0, 255)
        raw_image = self.image_qq.clone().squeeze(0).squeeze(0).detach().cpu().numpy()
        raw_image = np.clip(raw_image, a_min=0, a_max=255)
        return raw_image, loss.detach().cpu().numpy()
