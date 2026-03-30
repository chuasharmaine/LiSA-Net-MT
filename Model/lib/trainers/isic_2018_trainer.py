import os
import time
import numpy as np
import datetime

# try:
#     import nni
# except ImportError:
#     nni = None
import nni
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from lib import utils
from lib.metrics.ISIC2018 import ACCSEG, ACCCLS


class ISIC2018Trainer:
    """
    Trainer class for Segmentation-only, Classification-only, and Multitask
    """

    def __init__(self, opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric):

        self.opt = opt
        self.train_data_loader = train_loader
        self.valid_data_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.metric = metric
        self.best_metric_seg = 0.0
        self.best_metric_cls = 0.0
        self.device = opt["device"]
        self.seg_classes = opt.get("seg_classes", 1)
        self.cls_classes = opt.get("cls_classes", 7)
        if self.cls_classes is None:
            self.cls_classes = len(set([self.train_data_loader.dataset[i][1] for i in range(len(self.train_data_loader.dataset))]))

        self.seg_guided_cls = False
        self.seg_guided_cls = opt.get("seg_guided_cls", False)

        # Segmentation metrics
        if self.opt["segmentation"]:
            if "ACC_SEG" in self.opt["metric_names"]:
                self.metric["ACC_SEG"] = ACCSEG(num_classes=self.seg_classes, sigmoid_normalization=self.opt["sigmoid_normalization"])
            if "IoU" in self.opt["metric_names"]:
                from lib.metrics.ISIC2018 import IoU
                self.metric["IoU"] = IoU()
            if "DSC" in self.opt["metric_names"]:
                from lib.metrics.ISIC2018 import DICE
                self.metric["DSC"] = DICE()
            if "JI" in self.opt["metric_names"]:
                from lib.metrics.ISIC2018 import JI
                self.metric["JI"] = JI()

        # Classification metrics
        if self.opt["classification"]:
            self.cls_loss_function = self.loss_function["classification"]
            if "ACC_CLS" in self.opt["metric_names"]:
                self.metric["ACC_CLS"] = ACCCLS()
            if "F1_MACRO" in self.opt["metric_names"]:
                from lib.metrics.ISIC2018 import F1_MACRO
                self.metric["F1_MACRO"] = F1_MACRO()
            if "AUC_ROC" in self.opt["metric_names"]:
                from lib.metrics.ISIC2018 import AUC_ROC
                self.metric["AUC_ROC"] = AUC_ROC()

        if not self.opt["optimize_params"]:
            if self.opt["resume"] is None:
                self.execute_dir = os.path.join(opt["run_dir"], utils.datestr() + "_" + opt["model_name"] + "_" + opt["dataset_name"])
            else:
                self.execute_dir = os.path.dirname(os.path.dirname(self.opt["resume"]))
            self.checkpoint_dir = os.path.join(self.execute_dir, "checkpoints")
            self.tensorboard_dir = os.path.join(self.execute_dir, "board")
            self.log_txt_path = os.path.join(self.execute_dir, "log.txt")
            if self.opt["resume"] is None:
                utils.make_dirs(self.checkpoint_dir)
                utils.make_dirs(self.tensorboard_dir)
            utils.pre_write_txt("Complete the initialization of model:{}, optimizer:{}, and lr_scheduler:{}".format(self.opt["model_name"], self.opt["optimizer_name"], self.opt["lr_scheduler_name"]), self.log_txt_path)

        self.start_epoch = self.opt["start_epoch"]
        self.end_epoch = self.opt["end_epoch"]
        self.best_metric = opt["best_metric"]
        self.terminal_show_freq = opt["terminal_show_freq"]
        self.save_epoch_freq = opt["save_epoch_freq"]

        self.statistics_dict = self.init_statistics_dict()

    def training(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.reset_statistics_dict()

            self.optimizer.zero_grad()

            self.train_epoch(epoch)

            self.valid_epoch(epoch)

            train_class_IoU = np.nan_to_num(self.statistics_dict["train"].get("total_area_intersect", 0.0) / self.statistics_dict["train"].get("total_area_union", 1.0))
            valid_class_IoU = np.nan_to_num(self.statistics_dict["valid"].get("total_area_intersect", 0.0) / self.statistics_dict["valid"].get("total_area_union", 1.0))

            train_mean_IoU = np.mean(train_class_IoU) if self.opt["segmentation"] else 0
            valid_mean_IoU = np.mean(valid_class_IoU) if self.opt["segmentation"] else 0

            train_ACC_seg = self.statistics_dict["train"].get("ACC_seg_sum", 0.0) / max(1, self.statistics_dict["train"]["count"])
            valid_ACC_seg = self.statistics_dict["valid"].get("ACC_seg_sum", 0.0) / max(1, self.statistics_dict["valid"]["count"])

            train_ACC_cls = self.statistics_dict["train"].get("ACC_cls_sum", 0.0) / max(1, self.statistics_dict["train"]["count"])
            valid_ACC_cls = self.statistics_dict["valid"].get("ACC_cls_sum", 0.0) / max(1, self.statistics_dict["valid"]["count"])

            train_DSC = self.statistics_dict["train"].get("DSC_sum", 0.0) / max(1, self.statistics_dict["train"]["count"])
            valid_DSC = self.statistics_dict["valid"].get("DSC_sum", 0.0) / max(1, self.statistics_dict["valid"]["count"])

            train_JI = self.statistics_dict["train"].get("JI_sum", 0.0) / max(1, self.statistics_dict["train"]["count"])
            valid_JI = self.statistics_dict["valid"].get("JI_sum", 0.0) / max(1, self.statistics_dict["valid"]["count"])

            train_F1 = self.statistics_dict["train"].get("F1_MACRO", {}).get("avg", 0.0) / max(1, self.statistics_dict["train"]["count"])
            valid_F1 = self.statistics_dict["valid"].get("F1_MACRO", {}).get("avg", 0.0) / max(1, self.statistics_dict["valid"]["count"])

            valid_AUC = self.metric["AUC_ROC"].compute() if "AUC_ROC" in self.metric else 0.0

            seg_loss = self.statistics_dict["train"].get("seg_loss", 0.0) / max(1, self.statistics_dict["train"]["count"])
            cls_loss = self.statistics_dict["train"].get("cls_loss", 0.0) / max(1, self.statistics_dict["train"]["count"])

            if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(valid_JI)
            else:
                self.lr_scheduler.step()

            best_JI = self.best_metric_seg
            best_AUC = self.best_metric_cls

            if self.opt["segmentation"] and self.opt["classification"]:
                log_items = [
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch, self.end_epoch - 1,
                    self.optimizer.param_groups[0]['lr'],
                    self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                    train_DSC,
                    train_mean_IoU,
                    train_ACC_seg,
                    train_ACC_cls,
                    seg_loss,
                    cls_loss,
                    train_F1,
                    train_JI,
                    valid_DSC,
                    valid_mean_IoU,
                    valid_ACC_seg,
                    valid_ACC_cls,
                    valid_JI,
                    valid_F1,
                    valid_AUC,
                    best_JI,
                    best_AUC
                ]

                log_fmt = "[{}]  epoch:[{:05d}/{:05d}]  lr:{:.6f}  total_loss:{:.6f}  seg_loss:{:.6f}  cls_loss:{:.6f}  train_ACC_seg:{:.6f}  train_DSC:{:.6f}  train_IoU:{:.6f}  train_ACC_cls:{:.6f}  train_F1:{:.6f}  train_JI:{:.6f}  valid_ACC_seg:{:.6f}  valid_DSC:{:.6f}  valid_IoU:{:.6f}  valid_ACC_cls:{:.6f}  valid_F1:{:.6f}  valid_AUC:{:.6f}  valid_JI:{:.6f}  best_JI:{:.6f} best_AUC:{:.6f}"

            elif self.opt["classification"]:
                log_items = [
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch, self.end_epoch - 1,
                    self.optimizer.param_groups[0]['lr'],
                    self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                    train_ACC_cls,
                    train_F1,
                    valid_ACC_cls,
                    valid_F1,
                    valid_AUC,
                    best_AUC
                ]

                log_fmt = "[{}]  epoch:[{:05d}/{:05d}]  lr:{:.6f}  train_loss:{:.6f}  train_ACC_cls:{:.6f}  train_F1:{:.6f}  valid_ACC_cls:{:.6f}  valid_F1:{:.6f}  valid_AUC:{:.6f}  best_AUC:{:.6f}"
            else:
                log_items = [
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch, self.end_epoch - 1,
                    self.optimizer.param_groups[0]['lr'],
                    self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                    train_DSC,
                    train_mean_IoU,
                    train_ACC_seg,
                    train_JI,
                    valid_DSC,
                    valid_mean_IoU,
                    valid_ACC_seg,
                    valid_JI,
                    best_JI
                ]

                log_fmt = "[{}]  epoch:[{:05d}/{:05d}]  lr:{:.6f}  train_loss:{:.6f}  train_ACC_seg:{:.6f}  train_DSC:{:.6f}  train_IoU:{:.6f}  train_JI:{:.6f}  valid_ACC_seg:{:.6f} valid_DSC:{:.6f}  valid_IoU:{:.6f}   valid_JI:{:.6f}  best_JI:{:.6f}"
            
            log_str = log_fmt.format(*log_items)
            print(log_str)
            if not self.opt["optimize_params"]:
                utils.pre_write_txt(log_str, self.log_txt_path)

            if self.opt["optimize_params"]:
                nni.report_intermediate_result(valid_JI)

        if self.opt["optimize_params"]:
            nni.report_final_result(self.best_metric_cls if self.opt["classification"] else self.best_metric_seg)

    def train_epoch(self, epoch):

        self.model.train()

        for batch_idx, batch in enumerate(self.train_data_loader):

            input_tensor = seg_target = cls_target = None
            if len(batch) == 3:
                input_tensor, seg_target, cls_target = batch
            elif len(batch) == 2:
                if self.opt["segmentation"]:
                    input_tensor, seg_target = batch
                else:
                    input_tensor, cls_target = batch
            elif len(batch) == 1:
                input_tensor = batch[0]

            input_tensor = input_tensor.to(self.device)
            if seg_target is not None:
                seg_target = seg_target.to(self.device)
            if cls_target is not None:
                cls_target = cls_target.to(self.device)

            output = self.model(input_tensor)
            seg_out = None
            cls_out = None

            if isinstance(output, dict):
                seg_out = output.get("segmentation")
                cls_out = output.get("classification")
            elif isinstance(output, tuple):
                if self.opt["segmentation"]:
                    seg_out = output[0] if len(output) > 0 else None
                if self.opt["classification"]:
                    cls_out = output[1] if len(output) > 1 else None
            else:
                if self.opt["segmentation"]:
                    seg_out = output
                elif self.opt["classification"]:
                    cls_out = output

            # compute loss
            total_loss = None
            seg_loss_value = None
            cls_loss_value = None

            if seg_out is not None and seg_target is not None and "segmentation" in self.loss_function:
                seg_target = seg_target.long()
                seg_loss = self.loss_function["segmentation"](seg_out, seg_target)
                seg_loss_value = seg_loss.item()
                total_loss = seg_loss if total_loss is None else total_loss + seg_loss

            if cls_out is not None and cls_target is not None and "classification" in self.loss_function:
                cls_target_idx = cls_target.long()

                if self.seg_guided_cls and seg_out is not None:
                    seg_feat = torch.mean(seg_out, dim=(2,3))
                    if cls_out.ndim > 1 and seg_feat.shape[1] == cls_out.shape[1]:
                        cls_out = cls_out + 0.1 * seg_feat

                cls_loss = self.cls_loss_function(cls_out, cls_target_idx)
                cls_loss_value = cls_loss.item()
                total_loss = cls_loss if total_loss is None else total_loss + cls_loss

            if seg_loss_value is not None:
                self.statistics_dict["train"]["seg_loss"] += seg_loss_value * len(input_tensor)
            if cls_loss_value is not None:
                self.statistics_dict["train"]["cls_loss"] += cls_loss_value * len(input_tensor)

            # only backward if total_loss is a tensor
            if total_loss is not None:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            else:
                print("WARNING: no loss computed for this batch!")

            # update metrics
            if seg_out is not None and seg_target is not None:
                self.calculate_metric_and_update_statistcs(
                    seg_out.cpu(),
                    seg_target.cpu(),
                    len(input_tensor),
                    loss=total_loss.cpu() if total_loss is not None else None,
                    mode="train"
                )

            if cls_out is not None and cls_target is not None:
                self.calculate_metric_and_update_statistcs(
                    cls_out.cpu(),
                    cls_target_idx.cpu(),
                    len(input_tensor),
                    loss=total_loss.cpu() if total_loss is not None and (seg_out is None or seg_target is None) else None,
                    mode="train"
                )

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                self.log_training_progress(epoch, batch_idx)

    def valid_epoch(self, epoch):

        self.model.eval()

        if "AUC_ROC" in self.metric:
            self.metric["AUC_ROC"].reset()

        with torch.no_grad():

            for batch_idx, batch in enumerate(self.valid_data_loader):

                if self.opt["segmentation"] and self.opt["classification"]:
                    input_tensor, seg_target, cls_target = batch
                elif self.opt["segmentation"]:
                    input_tensor, seg_target = batch
                    cls_target = None
                elif self.opt["classification"]:
                    input_tensor, cls_target = batch
                    seg_target = None

                input_tensor = input_tensor.to(self.device)
                if seg_target is not None:
                    seg_target = seg_target.to(self.device)
                if cls_target is not None:
                    cls_target = cls_target.to(self.device)

                output = self.model(input_tensor)

                seg_out = None
                cls_out = None
                
                if isinstance(output, dict):
                    seg_out = output.get("segmentation")
                    cls_out = output.get("classification")
                elif isinstance(output, tuple):
                    if self.opt["segmentation"]:
                        seg_out = output[0] if len(output) > 0 else None
                    if self.opt["classification"]:
                        cls_out = output[1] if len(output) > 1 else None
                else:
                    if self.opt["segmentation"]:
                        seg_out = output
                    elif self.opt["classification"]:
                        cls_out = output

                if seg_out is not None and seg_target is not None:
                    if self.seg_classes == 1:
                        pred_prob = torch.sigmoid(seg_out)
                    else:
                        pred_prob = torch.softmax(seg_out, dim=1)

                    self.calculate_metric_and_update_statistcs(
                        pred_prob.cpu().float(),
                        seg_target.cpu().float(),
                        len(input_tensor),
                        loss=None,
                        mode="valid"
                    )

                if cls_out is not None and cls_out.numel() > 0 and cls_target is not None:
                    cls_target_idx = cls_target.long()

                    cls_prob = torch.softmax(cls_out, dim=1)

                    self.metric["AUC_ROC"].update(cls_prob.cpu(), cls_target_idx.cpu())

                    self.calculate_metric_and_update_statistcs(
                        cls_prob.cpu(),
                        cls_target_idx.cpu(),
                        len(input_tensor),
                        loss=None,
                        mode="valid"
                    )
            valid_count = self.statistics_dict["valid"]["count"]
            cur_JI = self.statistics_dict["valid"]["JI_sum"] / valid_count if valid_count > 0 else 0.0
            cur_AUC = self.metric["AUC_ROC"].compute()

            if cur_AUC > self.best_metric_cls and not self.opt["optimize_params"]:
                self.best_metric_cls = cur_AUC
                self.save(epoch, cur_AUC, self.best_metric_cls, type="best_cls")
            if cur_JI > self.best_metric_seg and not self.opt["optimize_params"]:
                self.best_metric_seg = cur_JI
                self.save(epoch, cur_JI, self.best_metric_seg, type="best_seg")

    def calculate_metric_and_update_statistcs(self, output, target, cur_batch_size, loss=None, mode="train"):
        # check which task
        is_seg_output = output.ndim == 4
        if is_seg_output:
            mask = torch.zeros(self.seg_classes)
        else:
            mask = torch.zeros(self.cls_classes)
        unique_index = torch.unique(target).int()
        for index in unique_index:
            if index < len(mask):
                mask[index] = 1
        self.statistics_dict[mode]["count"] += cur_batch_size
        for i, class_name in self.opt["index_to_class_dict"].items():
            if i < len(mask) and mask[i] == 1:
                self.statistics_dict[mode]["class_count"][class_name] += cur_batch_size
        if loss is not None and mode == "train":
            self.statistics_dict[mode]["loss"] += loss.item() * cur_batch_size
        for metric_name, metric_func in self.metric.items():
            if is_seg_output and metric_name in ["ACC_CLS", "F1_MACRO", "AUC_ROC"]:
                continue
            if (not is_seg_output) and metric_name in ["ACC_SEG", "DSC", "IoU", "JI"]:
                continue

            # segmentation metrics
            if metric_name == "IoU":
                area_intersect, area_union, _, _ = metric_func(output, target)
                area_intersect_np = area_intersect.numpy()
                area_union_np = area_union.numpy()

                if area_intersect_np.shape[0] != self.seg_classes:
                    tmp_intersect = np.zeros((self.seg_classes,), dtype=np.float32)
                    tmp_union = np.zeros((self.seg_classes,), dtype=np.float32)
                    length = min(area_intersect_np.shape[0], self.seg_classes)
                    tmp_intersect[:length] = area_intersect_np[:length]
                    tmp_union[:length] = area_union_np[:length]
                    area_intersect_np = tmp_intersect
                    area_union_np = tmp_union

                self.statistics_dict[mode]["total_area_intersect"] += area_intersect_np
                self.statistics_dict[mode]["total_area_union"] += area_union_np
            elif metric_name == "ACC_SEG":
                self.statistics_dict[mode]["ACC_seg_sum"] += metric_func(output, target) * cur_batch_size
            elif metric_name == "JI":
                batch_mean_JI = metric_func(output, target)
                self.statistics_dict[mode]["JI_sum"] += batch_mean_JI * cur_batch_size
            elif metric_name == "DSC":
                batch_mean_DSC = metric_func(output, target)
                self.statistics_dict[mode]["DSC_sum"] += batch_mean_DSC * cur_batch_size
            # classification metrics
            elif metric_name == "ACC_CLS":
                self.statistics_dict[mode]["ACC_cls_sum"] += metric_func(output, target) * cur_batch_size
            elif metric_name == "F1_MACRO":
                batch_f1 = metric_func(output, target)
                self.statistics_dict[mode]["F1_MACRO"]["avg"] += batch_f1 * cur_batch_size
            elif metric_name == "AUC_ROC":
                continue
            else:
                per_class_metric = metric_func(output, target)

                if not torch.is_tensor(per_class_metric):
                    per_class_metric = torch.tensor(per_class_metric)

                if per_class_metric.ndim == 0:
                    value = per_class_metric.item()
                    self.statistics_dict[mode][metric_name]["avg"] += value * cur_batch_size

                mask = mask.to(per_class_metric.device)
                mask = mask[:len(per_class_metric)]
                per_class_metric = per_class_metric[:len(mask)] * mask

                self.statistics_dict[mode][metric_name]["avg"] += (torch.sum(per_class_metric) / torch.sum(mask)).item() * cur_batch_size

                for j, class_name in self.opt["index_to_class_dict"].items():
                    self.statistics_dict[mode][metric_name][class_name] += per_class_metric[j].item() * cur_batch_size

    def init_statistics_dict(self):
        statistics_dict = {
            "train": {
                metric_name: {class_name: 0.0 for _, class_name in self.opt["index_to_class_dict"].items()}
                for metric_name in self.opt["metric_names"]
            },
            "valid": {
                metric_name: {class_name: 0.0 for _, class_name in self.opt["index_to_class_dict"].items()}
                for metric_name in self.opt["metric_names"]
            }
        }
        seg_stat_classes = self.seg_classes if (self.opt["segmentation"] and self.seg_classes is not None) else 0
        statistics_dict["train"]["total_area_intersect"] = np.zeros((seg_stat_classes,))
        statistics_dict["train"]["total_area_union"] = np.zeros((seg_stat_classes,))
        statistics_dict["valid"]["total_area_intersect"] = np.zeros((seg_stat_classes,))
        statistics_dict["valid"]["total_area_union"] = np.zeros((seg_stat_classes,))
        statistics_dict["train"]["JI_sum"] = 0.0
        statistics_dict["valid"]["JI_sum"] = 0.0
        statistics_dict["train"]["ACC_seg_sum"] = 0.0
        statistics_dict["train"]["ACC_cls_sum"] = 0.0
        statistics_dict["valid"]["ACC_seg_sum"] = 0.0
        statistics_dict["valid"]["ACC_cls_sum"] = 0.0
        statistics_dict["train"]["DSC_sum"] = 0.0
        statistics_dict["valid"]["DSC_sum"] = 0.0
        for metric_name in self.opt["metric_names"]:
            statistics_dict["train"][metric_name]["avg"] = 0.0
            statistics_dict["valid"][metric_name]["avg"] = 0.0
        statistics_dict["train"]["loss"] = 0.0
        statistics_dict["train"]["seg_loss"] = 0.0
        statistics_dict["train"]["cls_loss"] = 0.0
        statistics_dict["train"]["class_count"] = {class_name: 0 for _, class_name in self.opt["index_to_class_dict"].items()}
        statistics_dict["valid"]["class_count"] = {class_name: 0 for _, class_name in self.opt["index_to_class_dict"].items()}
        statistics_dict["train"]["count"] = 0
        statistics_dict["valid"]["count"] = 0

        return statistics_dict

    def reset_statistics_dict(self):
        seg_stat_classes = self.seg_classes if (self.opt["segmentation"] and self.seg_classes is not None) else 0
        for phase in ["train", "valid"]:
            self.statistics_dict[phase]["count"] = 0
            self.statistics_dict[phase]["total_area_intersect"] = np.zeros((seg_stat_classes,))
            self.statistics_dict[phase]["total_area_union"] = np.zeros((seg_stat_classes,))
            self.statistics_dict[phase]["JI_sum"] = 0.0
            self.statistics_dict[phase]["ACC_seg_sum"] = 0.0
            self.statistics_dict[phase]["ACC_cls_sum"] = 0.0
            self.statistics_dict[phase]["DSC_sum"] = 0.0
            for _, class_name in self.opt["index_to_class_dict"].items():
                self.statistics_dict[phase]["class_count"][class_name] = 0
            if phase == "train":
                self.statistics_dict[phase]["loss"] = 0.0
                self.statistics_dict[phase]["seg_loss"] = 0.0
                self.statistics_dict[phase]["cls_loss"] = 0.0
            for metric_name in self.opt["metric_names"]:
                self.statistics_dict[phase][metric_name]["avg"] = 0.0
                for _, class_name in self.opt["index_to_class_dict"].items():
                    self.statistics_dict[phase][metric_name][class_name] = 0.0

    def save(self, epoch, metric, best_metric, type="normal"):
        state = {
            "epoch": epoch,
            "best_metric": best_metric,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
        }
        if type == "normal":
            save_filename = "{:04d}_{}_{:.4f}.state".format(epoch, self.opt["model_name"], metric)
        else:
            save_filename = '{}_{}.state'.format(type, self.opt["model_name"])
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(state, save_path)
        if type == "normal":
            save_filename = "{:04d}_{}_{:.4f}.pth".format(epoch, self.opt["model_name"], metric)
        else:
            save_filename = '{}_{}.pth'.format(type, self.opt["model_name"])
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(self.model.state_dict(), save_path)

    def load(self):
        if self.opt["resume"] is not None:
            if self.opt["pretrain"] is None:
                raise RuntimeError("Training weights must be specified to continue training")

            resume_state_dict = torch.load(self.opt["resume"], map_location=lambda storage, loc: storage.cuda(self.device))
            self.start_epoch = resume_state_dict["epoch"] + 1
            self.best_metric = resume_state_dict["best_metric"]
            self.optimizer.load_state_dict(resume_state_dict["optimizer"])
            self.lr_scheduler.load_state_dict(resume_state_dict["lr_scheduler"])

            pretrain_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
            model_state_dict = self.model.state_dict()
            load_count = 0
            for param_name in model_state_dict.keys():
                if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                    model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                    load_count += 1
            self.model.load_state_dict(model_state_dict, strict=True)
            print("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)))
            if not self.opt["optimize_params"]:
                utils.pre_write_txt("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)), self.log_txt_path)
        else:
            if self.opt["pretrain"] is not None:
                pretrain_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
                model_state_dict = self.model.state_dict()
                load_count = 0
                for param_name in model_state_dict.keys():
                    if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                        model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                        load_count += 1
                self.model.load_state_dict(model_state_dict, strict=True)
                print("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)))
                if not self.opt["optimize_params"]:
                    utils.pre_write_txt("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)), self.log_txt_path)

    def log_training_progress(self, epoch, batch_idx):
        avg_total_loss = self.statistics_dict['train']['loss'] / max(1, self.statistics_dict['train']['count'])
        avg_seg_loss = self.statistics_dict['train'].get('seg_loss', 0.0) / max(1, self.statistics_dict['train']['count'])
        avg_cls_loss = self.statistics_dict['train'].get('cls_loss', 0.0) / max(1, self.statistics_dict['train']['count'])

        if self.opt["segmentation"] and self.opt["classification"]:
            log_str = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]  epoch:[{epoch:05d}/{self.end_epoch-1:05d}]  step:[{batch_idx+1:04d}/{len(self.train_data_loader):04d}]  lr:{self.optimizer.param_groups[0]['lr']:.6f}  total_loss:{avg_total_loss:.6f}  seg_loss:{avg_seg_loss:.6f}  cls_loss:{avg_cls_loss:.6f}"
        else:
            log_str = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]  epoch:[{epoch:05d}/{self.end_epoch-1:05d}]  step:[{batch_idx+1:04d}/{len(self.train_data_loader):04d}]  lr:{self.optimizer.param_groups[0]['lr']:.6f}  loss:{avg_total_loss:.6f}"

        print(log_str)
        if not self.opt["optimize_params"]:
            utils.pre_write_txt(log_str, self.log_txt_path)