import argparse
import json
import random
import re
import string
from datetime import datetime, timedelta
from pathlib import Path
import config


# ============================================================
# 配置
# ============================================================

DATA_DISTRIBUTION = {
    "simple": 0.20,
    "nested": 0.25,
    "deep_nested": 0.20,
    "array": 0.15,
    "nested_array": 0.10,
    "complex": 0.10,
}


# ============================================================
# 基础随机数据
# ============================================================

FIRST_NAMES = [
    "John", "Michael", "David", "James", "Robert",
    "William", "Daniel", "Thomas", "Christopher", "Matthew",
    "Sarah", "Emma", "Olivia", "Sophia", "Emily",
    "Jessica", "Alice", "Laura", "Anna", "Jennifer",
]

LAST_NAMES = [
    "Smith", "Johnson", "Brown", "Taylor", "Wilson",
    "Miller", "Davis", "Anderson", "Thomas", "Moore",
    "Martin", "Jackson", "White", "Harris", "Clark",
]

COMPANIES = [
    "Acme Corporation",
    "BlueTech Ltd",
    "Northwind Trading",
    "Global Components Ltd",
    "Summit Industries",
    "Evergreen Solutions",
    "Pioneer Logistics",
    "Silverline Systems",
    "Vertex Technologies",
    "Continental Supplies",
]

CITIES = [
    "London",
    "Manchester",
    "Birmingham",
    "Liverpool",
    "Bristol",
    "Leeds",
    "Berlin",
    "Munich",
    "Paris",
    "Amsterdam",
]

COUNTRIES = [
    "UK",
    "Germany",
    "France",
    "Netherlands",
    "United States",
]

PRODUCTS = [
    "network router",
    "network switch",
    "firewall appliance",
    "wireless access point",
    "server rack",
    "power adapter",
    "Ethernet cable",
    "industrial controller",
    "barcode scanner",
    "storage device",
]

PAYMENT_METHODS = [
    "bank transfer",
    "credit card",
    "PayPal",
    "wire transfer",
    "direct debit",
]

ORDER_STATUSES = [
    "pending",
    "confirmed",
    "processing",
    "shipped",
    "delivered",
    "cancelled",
]

ROLES = [
    "Purchasing Manager",
    "Finance Manager",
    "Sales Manager",
    "Operations Manager",
    "Project Manager",
    "Account Manager",
]


# ============================================================
# 基础工具
# ============================================================

def random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def random_email(name=None):
    if name is None:
        name = random_name()

    first, last = name.lower().split()
    company = random.choice([
        "example.com",
        "acme.com",
        "bluetech.com",
        "northwind.com",
        "globalparts.com",
    ])

    return f"{first}.{last}@{company}"


def random_phone():
    return f"+44 {random.randint(10, 99)} {random.randint(1000, 9999)} {random.randint(1000, 9999)}"


def random_company():
    return random.choice(COMPANIES)


def random_id(prefix, length=5):
    chars = string.ascii_uppercase + string.digits
    return prefix + "-" + "".join(random.choices(chars, k=length))


def random_date():
    start = datetime(2026, 1, 1)
    date = start + timedelta(days=random.randint(0, 365))
    return date.strftime("%Y-%m-%d")


def random_amount():
    return round(random.uniform(100, 50000), 2)


def random_address():
    street_number = random.randint(1, 200)
    street_names = [
        "King Street",
        "Market Street",
        "High Street",
        "Victoria Road",
        "Station Road",
        "Church Street",
        "London Road",
    ]

    return (
        f"{street_number} {random.choice(street_names)}, "
        f"{random.choice(CITIES)}"
    )


def random_product():
    return random.choice(PRODUCTS)


def random_quantity():
    return random.randint(1, 100)


# ============================================================
# instruction
# ============================================================

INSTRUCTION_TEMPLATES = [
    "Extract the {fields} from the email.",
    "Identify the {fields} mentioned in the email.",
    "Find the {fields} in the following email.",
    "Extract and organize the {fields} into JSON.",
    "Parse the email and return the {fields}.",
    "Please extract the {fields} from the message.",
    "Retrieve the {fields} described in the email.",
    "Identify and structure the {fields} found in the message.",
]


def make_instruction(field_description):
    template = random.choice(INSTRUCTION_TEMPLATES)

    return template.format(
        fields=field_description
    )


# ============================================================
# 邮件噪声
# ============================================================

OPENINGS = [
    "Hi Team,",
    "Dear Team,",
    "Hello everyone,",
    "Good morning,",
    "Dear Customer Service,",
    "Hello,",
]

CLOSINGS = [
    "Best regards,",
    "Kind regards,",
    "Best wishes,",
    "Regards,",
    "Thank you,",
]

SIGNATURES = [
    "John Smith\nSales Department",
    "Sarah Miller\nCustomer Operations",
    "Michael Brown\nAccount Management",
    "Emma Wilson\nFinance Department",
    "Daniel Taylor\nLogistics Team",
]

DISCLAIMER = (
    "This email and any attachments are confidential and intended "
    "only for the recipient."
)


def add_email_wrapper(body, add_disclaimer=False):
    parts = [
        random.choice(OPENINGS),
        "",
        body,
        "",
        random.choice(CLOSINGS),
        random.choice(SIGNATURES),
    ]

    if add_disclaimer:
        parts.extend([
            "",
            DISCLAIMER,
        ])

    return "\n".join(parts)


# ============================================================
# Simple
# ============================================================

def generate_simple():
    name = random_name()
    company = random_company()
    email = random_email(name)
    order_id = random_id("ORD")
    amount = random_amount()

    choices = [
        ("customer_name", name, f"the customer name"),
        ("company_name", company, f"the company name"),
        ("email", email, f"the email address"),
        ("order_id", order_id, f"the order ID"),
        ("order_amount", amount, f"the order amount"),
    ]

    selected = random.sample(choices, random.randint(1, 3))

    target = {}
    sentences = []

    for key, value, description in selected:
        target[key] = value

        if key == "customer_name":
            sentences.append(f"The customer is {value}.")
        elif key == "company_name":
            sentences.append(f"The company is {value}.")
        elif key == "email":
            sentences.append(f"You can reach us at {value}.")
        elif key == "order_id":
            sentences.append(f"The order reference is {value}.")
        elif key == "order_amount":
            sentences.append(f"The total order value is USD {value:.2f}.")

    body = " ".join(sentences)

    field_names = [description for _, _, description in selected]

    instruction = make_instruction(
        ", ".join(field_names)
    )

    return instruction, add_email_wrapper(body), target


# ============================================================
# Nested
# ============================================================

def generate_nested():
    name = random_name()
    company = random_company()

    target = {
        "customer": {
            "name": name,
            "company": company,
        }
    }

    body = (
        f"I am contacting you regarding {name} from {company}. "
        f"We would like to update the customer information in your system."
    )

    if random.random() < 0.7:
        email = random_email(name)

        target["customer"]["contact"] = {
            "email": email
        }

        body += f" The customer's email address is {email}."

    if random.random() < 0.6:
        phone = random_phone()

        # 上面的 contact 分支是独立随机事件，可能未执行，
        # 因此这里要用 setdefault 兜底，避免 KeyError: 'contact'
        target["customer"].setdefault("contact", {})["phone"] = phone

        body += f" The direct telephone number is {phone}."

    if random.random() < 0.5:
        target["customer"]["address"] = {
            "city": random.choice(CITIES),
            "country": random.choice(COUNTRIES),
        }

        body += (
            f" The customer is located in "
            f"{target['customer']['address']['city']}, "
            f"{target['customer']['address']['country']}."
        )

    instruction = make_instruction(
        "the customer information"
    )

    return instruction, add_email_wrapper(body), target


# ============================================================
# Deep nested
# ============================================================

def generate_deep_nested():
    name = random_name()
    company = random_company()

    city = random.choice(CITIES)
    country = random.choice(COUNTRIES)

    target = {
        "customer": {
            "name": name,
            "organization": {
                "name": company,
                "location": {
                    "city": city,
                    "country": country,
                }
            }
        }
    }

    body = (
        f"Our customer {name} represents {company}. "
        f"The company operates from {city}, {country}."
    )

    if random.random() < 0.8:
        email = random_email(name)

        target["customer"]["contact"] = {
            "email": email
        }

        body += f" The primary contact email is {email}."

    if random.random() < 0.6:
        phone = random_phone()

        # 上面的 contact 分支是独立随机事件，可能未执行，
        # 因此这里要用 setdefault 兜底，避免 KeyError: 'contact'
        target["customer"].setdefault("contact", {})["phone"] = phone

        body += f" The phone number is {phone}."

    if random.random() < 0.5:
        target["customer"]["account"] = {
            "id": random_id("ACC"),
            "status": random.choice(
                ["active", "inactive", "pending"]
            )
        }

        body += (
            f" The account reference is "
            f"{target['customer']['account']['id']}."
        )

    instruction = make_instruction(
        "the detailed customer information"
    )

    return instruction, add_email_wrapper(body), target


# ============================================================
# Array
# ============================================================

def generate_array():
    customer = random_name()
    company = random_company()

    item_count = random.randint(2, 5)

    items = []

    sentences = []

    for _ in range(item_count):
        product = random_product()
        quantity = random_quantity()

        items.append({
            "product": product,
            "quantity": quantity,
        })

        sentences.append(
            f"{quantity} units of {product}"
        )

    order_id = random_id("ORD")

    target = {
        "order": {
            "id": order_id,
            "customer": customer,
            "items": items,
        }
    }

    body = (
        f"{customer} from {company} placed order {order_id}. "
        f"The order contains " +
        ", ".join(sentences) +
        "."
    )

    instruction = make_instruction(
        "the order, customer, and ordered items"
    )

    return instruction, add_email_wrapper(body), target


# ============================================================
# Nested array
# ============================================================

def generate_nested_array():
    company = random_company()

    contact_count = random.randint(2, 4)

    contacts = []

    contact_sentences = []

    for _ in range(contact_count):
        name = random_name()
        role = random.choice(ROLES)
        email = random_email(name)

        contacts.append({
            "name": name,
            "role": role,
            "contact": {
                "email": email
            }
        })

        contact_sentences.append(
            f"{name} ({role}, {email})"
        )

    target = {
        "organization": {
            "name": company,
            "contacts": contacts,
        }
    }

    body = (
        f"{company} provided the following contacts: "
        + "; ".join(contact_sentences)
        + "."
    )

    instruction = make_instruction(
        "the organization and its contacts"
    )

    return instruction, add_email_wrapper(body), target


# ============================================================
# Complex
# ============================================================

def generate_complex():
    customer_name = random_name()
    company = random_company()

    order_count = random.randint(1, 3)

    orders = []

    body_parts = []

    for _ in range(order_count):
        order_id = random_id("ORD")

        item_count = random.randint(1, 3)

        items = []

        item_sentences = []

        for _ in range(item_count):
            product = random_product()
            quantity = random_quantity()

            items.append({
                "product": product,
                "quantity": quantity,
                "pricing": {
                    "unit_price": round(
                        random.uniform(10, 2000), 2
                    ),
                    "currency": random.choice(
                        ["USD", "EUR", "GBP"]
                    )
                }
            })

            item_sentences.append(
                f"{quantity} {product}"
            )

        order = {
            "order_id": order_id,
            "status": random.choice(ORDER_STATUSES),
            "items": items,
            "payment": {
                "method": random.choice(PAYMENT_METHODS),
                "status": random.choice(
                    ["paid", "pending", "failed"]
                )
            }
        }

        orders.append(order)

        body_parts.append(
            f"Order {order_id} contains "
            + ", ".join(item_sentences)
            + f" and is currently {order['status']}."
        )

    target = {
        "customer": {
            "name": customer_name,
            "company": company,
            "contact": {
                "email": random_email(customer_name),
                "phone": random_phone(),
            }
        },
        "orders": orders,
        "shipping": {
            "address": random_address(),
            "requested_date": random_date(),
        }
    }

    body = (
        f"{customer_name} from {company} contacted us about "
        f"several orders. "
        + " ".join(body_parts)
        + f" The requested delivery date is "
        f"{target['shipping']['requested_date']}."
    )

    instruction = make_instruction(
        "the customer, order, payment, and shipping information"
    )

    return instruction, add_email_wrapper(
        body,
        add_disclaimer=random.random() < 0.3
    ), target


# ============================================================
# 类型选择
# ============================================================

GENERATORS = {
    "simple": generate_simple,
    "nested": generate_nested,
    "deep_nested": generate_deep_nested,
    "array": generate_array,
    "nested_array": generate_nested_array,
    "complex": generate_complex,
}


def calculate_counts(num_samples):
    """
    使用最大余数法，把总数量按照固定比例分配。
    保证最终数量严格等于 num_samples。
    """

    raw_counts = {
        name: num_samples * ratio
        for name, ratio in DATA_DISTRIBUTION.items()
    }

    counts = {
        name: int(value)
        for name, value in raw_counts.items()
    }

    remaining = num_samples - sum(counts.values())

    remainders = sorted(
        DATA_DISTRIBUTION.keys(),
        key=lambda name: (
            raw_counts[name] - counts[name]
        ),
        reverse=True
    )

    for i in range(remaining):
        counts[remainders[i]] += 1

    return counts


# ============================================================
# 生成数据
# ============================================================

def generate_dataset(num_samples, output_file, seed=42):
    random.seed(seed)

    counts = calculate_counts(num_samples)

    print("=" * 60)
    print(f"Generating {num_samples} samples")
    print("=" * 60)

    for name, count in counts.items():
        ratio = DATA_DISTRIBUTION[name]

        print(
            f"{name:15s}: "
            f"{count:6d} "
            f"({ratio:.0%})"
        )

    print("=" * 60)

    output_path = Path(output_file)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    samples = []

    # --------------------------------------------------------
    # 按比例生成
    # --------------------------------------------------------

    for data_type, count in counts.items():

        generator = GENERATORS[data_type]

        for _ in range(count):

            instruction, email, target = generator()

            # target 序列化为紧凑的标准 JSON 格式（包含大括号与双引号）
            target_text = json.dumps(
                target,
                ensure_ascii=False,
                separators=(",", ":")
            )

            sample = {
                "instruction": instruction,
                "input": email,
                "target": target_text,
            }

            # ----------------------------------------------------
            # 生成的数据不允许包含换行符（\r\n / \n）
            #
            # add_email_wrapper 用 "\n" 拼装邮件正文，
            # 这里统一把所有字符串字段中的换行符连同后续空格删除，
            # 与 process.py 的 remove_newlines 逻辑保持一致。
            # ----------------------------------------------------
            sample = {
                k: re.sub(r"\r?\n\s*", "", v) if isinstance(v, str) else v
                for k, v in sample.items()
            }

            samples.append(sample)

    # --------------------------------------------------------
    # 整体再次随机打乱
    # --------------------------------------------------------

    random.shuffle(samples)

    # --------------------------------------------------------
    # 写 JSONL
    #
    # newline="":
    #     禁用 Windows 文本模式换行转换，
    #     避免行分隔符 \n 被写成 \r\n。
    #
    # 整行用标准 JSON 序列化：
    #     target 是 JSON 对象，直接写成 {"customer":{...}}，
    #     无引号包裹、无转义，可被 load_dataset("json") 解析。
    # --------------------------------------------------------

    with output_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        for sample in samples:

            f.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                    separators=(",", ":")
                )
                + "\n"
            )

    print(f"\nSaved to: {output_path}")
    print(f"Total samples: {len(samples)}")


# ============================================================
# main
# ============================================================

def main():

    generate_dataset(
        num_samples=3000,
        output_file=config.DATA_DIR / "generated_data.jsonl",
        seed=42,
    )


if __name__ == "__main__":
    main()