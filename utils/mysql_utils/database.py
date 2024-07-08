from pipes import quote

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 对密码进行URL编码
password = quote('123456')

# 连接MYSQL
SQLALCHEMY_DATABASE_URL = f'mysql+pymysql://root:{password}@localhost:3306/summerterm'

engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=True)

# 创建基本的映射类
Base = declarative_base(name='Base')


