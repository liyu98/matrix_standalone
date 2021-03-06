# Server（联邦学习服务端）
# Perform model aggregation on the local model uploaded by the selected client.

import models, torch


class Server(object):
    # Define the constructor to complete the initialization of configuration parameters
    def __init__(self, conf, eval_dataset):
        self.conf = conf
        self.global_model = models.get_model(self.conf["model_name"])
        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    # global aggregation model
    # weight_accumulator stores the upload parameter change value/difference value of each client
    def model_aggregate(self, weight_accumulator):
        # Traverse the server's global model
        for name, data in self.global_model.state_dict().items():
            # update each layer multiplied by the learning rate 更新每一层乘上学习率
            update_per_layer = weight_accumulator[name] * self.conf["lambda"]
            # cumulative sum
            if data.type() != update_per_layer.type():
                # Because the type of update_per_layer is floatTensor, it will be converted to
                # LongTensor of the model (with a certain precision loss)
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    # evaluate function
    def model_eval(self):
        # Enable model evaluation mode (without modifying parameters)
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        # Iterate over the evaluation data set
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            # 获取所有的样本总量大小
            dataset_size += data.size()[0]
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = self.global_model(data)
            # 聚合所有的损失 sum up batch loss
            # cross_entropy 交叉熵函数计算损失
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            # 统计预测结果与真实标签target的匹配总个数
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        # Calculate accuracy
        acc = 100.0 * (float(correct) / float(dataset_size))
        print("server acc", acc)
        # Calculate the loss value
        total_l = total_loss / dataset_size

        return acc, total_l
