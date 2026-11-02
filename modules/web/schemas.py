"""浏览器与服务端之间的受限协议，不接受任意本地磁盘路径。"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageRef(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    kind: Literal['upload', 'archived']
    id: str = Field(min_length=1, max_length=120, pattern=r'^[A-Za-z0-9_-]+$')


class MessageRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    question: str = Field(default='', max_length=100000)
    request_id: str = Field(min_length=1, max_length=120, pattern=r'^[A-Za-z0-9_-]+$')
    images: list[ImageRef] = Field(default_factory=list, max_length=100)

    @model_validator(mode='after')
    def check_message(self):
        self.question = self.question.strip()
        if not self.question:
            if not self.images:
                raise ValueError('请输入分析需求或选择航拍图片')
            # 默认文案只要求分析，不隐含要求生成报告。
            self.question = '分析所选灾情图像'
        refs = [(image.kind, image.id) for image in self.images]
        if len(set(refs)) != len(refs):
            raise ValueError('同一消息不能重复引用同一附件')
        return self


class SettingsPatch(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    version: str
    changes: dict
