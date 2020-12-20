from flask import Flask, request

app = Flask(__name__)


@app.route("/index", methods=["GET", "POST"])
def index():
    # request中包含了前端发来的所有请求数据
    # request.form可以直接提取请求体中的表达格式的数据，是一个类字典的对象
    # request.form["name"] #通常不是有这种方式提取参数，因为如果前端没有传这参数程序就会报错,为了程序健壮性通常使用get方式获取
    # 通过get方法只能拿到多个重名参数的第一个
    name = request.form.get("name")
    age = request.form.get("country")
    # 获取同名参数,将名为name的参数全部提取到一个列表中
    name_li = request.form.getlist("name")
    # 提出url中的参数（查询字符串）
    name = request.args.get("name")
    age=request.args.get("age")
    print(request.data)
    return ("name:{},age:{}".format(name,age))


if __name__ == '__main__':
    #127.0.0.1:80/index?name=dongli&age=24
    app.run(port=80, debug=True)