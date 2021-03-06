import smtplib
import socket
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import time

from config import c


class Notifier(object):
    def __init__(self):
        self.hostname = socket.gethostname()
        self.mail_host = 'smtp.163.com'
        # 163用户名
        self.mail_user = 'jixiyang_hit'
        # 密码(部分邮箱为授权码)
        self.mail_pass = '696227jXY'
        # 邮件发送方邮箱地址
        self.sender = 'jixiyang_hit@163.com'
        # 邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发
        self.receivers = ['1044456468@qq.com']
        self.smtpObj = smtplib.SMTP()

    def login(self):
        # 连接到服务器
        print("smtp connection")
        self.smtpObj.connect(self.mail_host, 25)
        # 登录到服务器
        print("smtp logging")
        self.smtpObj.login(self.mail_user, self.mail_pass)
        print("smtp config complete")

    def close(self):
        self.smtpObj.quit()
        print("smtp connection closed")

    def send(self, content):
        self.login()
        # 设置email信息
        # 邮件内容设置
        content += "\n" + "config file path: "+c.SAVE_PATH
        message = MIMEText(content, 'plain', 'utf-8')
        # 邮件主题
        message['Subject'] = self.hostname
        # 发送方信息
        message['From'] = self.sender
        # 接受方信息
        message['To'] = self.receivers[0]

        # 登录并发送邮件
        try:
            self.smtpObj.sendmail(
                self.sender, self.receivers, message.as_string())
            # 退出
            print('success')
        except smtplib.SMTPException as e:
            print('error', e)  # 打印错误
        self.close()

    def eval(self, step, img_path):
        self.login()
        # 邮件内容设置
        message = MIMEMultipart()
        # 邮件主题
        message['Subject'] = self.hostname + f" in {step}"
        # 发送方信息
        message['From'] = self.hostname + f"<{self.sender}>"
        # 接受方信息
        message['To'] = "thanatos" + f"<{self.receivers[0]}>"
        content = self.hostname + "<br>config file path: " + c.SAVE_PATH + f"<br>iter{step}<br>"
        content = '<html><body><p>' + content + '</p>' +'<p><img src="cid:0"></p>' +'</body></html>'
        txt = MIMEText(content, 'html', 'utf-8')
        message.attach(txt)
        # file = open('/Users/thanatos/Pictures/图片/杂乱/IMG_9782.JPG', 'rb')
        # img_data = file.read()
        # file.close()
        # img = MIMEImage(img_data)
        # img.add_header('Content-ID', 'dns_config')
        # message.attach(img)

        with open(img_path, 'rb') as f:
            # 设置附件的MIME和文件名，这里是png类型:
            mime = MIMEBase('image', 'png', filename='result.png')
            # 加上必要的头信息:
            mime.add_header('Content-Disposition', 'attachment', filename='test.png')
            mime.add_header('Content-ID', '<0>')
            mime.add_header('X-Attachment-Id', '0')
            # 把附件的内容读进来:
            mime.set_payload(f.read())
            # 用Base64编码:
            encoders.encode_base64(mime)
            # 添加到MIMEMultipart:
            message.attach(mime)
        # 登录并发送邮件
        try:
            self.smtpObj.sendmail(
                self.sender, self.receivers, message.as_string())
            # 退出
            print('success')
        except smtplib.SMTPException as e:
            print('error', e)  # 打印错误

        message.attach(MIMEText(content, 'plain', 'utf-8'))
        self.close()


if __name__ == '__main__':
    no = Notifier()
    no.eval(14999, '/extend/gru_tf_data/gru_tf_grad/Metrics/104999/average.jpg')
